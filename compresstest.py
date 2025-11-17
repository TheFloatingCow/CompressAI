# pip install compressai pillow matplotlib numpy torch

import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import os

from PIL import Image

from compressai.zoo import (
    bmshj2018_factorized,
    bmshj2018_hyperprior,
    cheng2020_anchor,
    mbt2018,
    mbt2018_mean,
)

# 1. Load an image (must be RGB)
img = Image.open(r"c:\Users\adamm\Downloads\424\test2.png").convert("RGB")

# Resize image to dimensions divisible by 16 (required by the models)
h, w = img.size
h = (h // 16) * 16
w = (w // 16) * 16
img = img.resize((h, w), Image.Resampling.LANCZOS)

# Convert to tensor [1, 3, H, W], normalized to [0,1]
x = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0

# Check device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
x = x.to(device)

# Define all available models
models_info = [
    ('bmshj2018-factorized', lambda: bmshj2018_factorized(quality=3, pretrained=True).eval().to(device)),
    ('bmshj2018-hyperprior', lambda: bmshj2018_hyperprior(quality=3, pretrained=True).eval().to(device)),
    ('mbt2018-mean', lambda: mbt2018_mean(quality=3, pretrained=True).eval().to(device)),
    ('mbt2018', lambda: mbt2018(quality=3, pretrained=True).eval().to(device)),
    ('cheng2020-anchor', lambda: cheng2020_anchor(quality=3, pretrained=True).eval().to(device)),
]

# Store results
results = []
reconstructions = []


# Helper: robustly compute total bytes for the compressed "strings" structure
def compute_total_bytes(strings):
    """Return total number of bytes contained in `strings`.

    `strings` may be a list of bytes objects, or a list of lists/tuples
    containing bytes. This function handles both shapes and falls back
    gracefully if elements aren't bytes-like.
    """
    total = 0
    if not strings:
        return 0
    for s in strings:
        # If this element is bytes-like, count its length
        if isinstance(s, (bytes, bytearray)):
            total += len(s)
            continue

        # If this element is a list/tuple, iterate its items
        if isinstance(s, (list, tuple)):
            for item in s:
                if isinstance(item, (bytes, bytearray)):
                    total += len(item)
                else:
                    # fallback: try to use len() if possible
                    try:
                        total += len(item)
                    except Exception:
                        # ignore items we can't measure
                        pass
            continue

        # Last resort: try len() on the element
        try:
            total += len(s)
        except Exception:
            pass
    return total


def save_compressed_bytes(strings, filepath):
    """Write compressed `strings` to `filepath` as raw bytes.

    Returns the number of bytes written. The function handles nested
    lists/tuples of bytes as returned by different CompressAI versions.
    """
    total_written = 0
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        if not strings:
            return 0
        for s in strings:
            if isinstance(s, (bytes, bytearray)):
                f.write(s)
                total_written += len(s)
                continue

            if isinstance(s, (list, tuple)):
                for item in s:
                    if isinstance(item, (bytes, bytearray)):
                        f.write(item)
                        total_written += len(item)
                    else:
                        try:
                            # try writing memoryview/other buffer-like
                            f.write(bytes(item))
                            total_written += len(item)
                        except Exception:
                            pass
                continue

            try:
                f.write(bytes(s))
                total_written += len(s)
            except Exception:
                # ignore items we can't convert
                pass
    return total_written

# Test each model
for model_name, model_fn in models_info:
    try:
        print(f"Testing {model_name}...")
        model = model_fn().eval()
        
        # Compress and decompress
        out = model.compress(x)
        recon = model.decompress(out["strings"], out["shape"])["x_hat"].clamp(0, 1)

        # Compute PSNR
        mse = torch.mean((x - recon).pow(2))
        psnr = 10 * torch.log10(1 / mse)

        # Compute compressed size (in bytes) from the returned byte-strings
        # Use a robust helper since `out["strings"]` can be a list of
        # bytes objects or a list of lists/tuples of bytes.
        total_bytes = compute_total_bytes(out.get("strings"))

        # Save compressed stream to a .bin file for inspection/download
        script_dir = os.path.dirname(os.path.abspath(__file__))
        safe_name = model_name.replace('/', '_').replace(' ', '_')
        out_path = os.path.join(script_dir, f"{safe_name}.bin")
        try:
            written = save_compressed_bytes(out.get("strings"), out_path)
            # If our saved-bytes differs from computed, prefer the actual file size
            if written and written != total_bytes:
                total_bytes = written
        except Exception:
            written = 0

        total_bits = total_bytes * 8
        original_bits = x.numel() * 8  # assuming 8-bit original per channel
        compression_ratio = original_bits / total_bits if total_bits > 0 else 0
        compressed_kb = total_bytes / 1024.0

        results.append({
            "name": model_name,
            "psnr": psnr.item(),
            "compression_ratio": compression_ratio,
            "bpp": total_bits / (x.shape[2] * x.shape[3]) if total_bits > 0 else 0,
            "compressed_bytes": int(total_bytes),
            "compressed_kb": float(compressed_kb),
        })
        
        # Move reconstruction to CPU and convert to numpy image [H, W, C]
        recon_np = recon.squeeze().permute(1, 2, 0).detach().cpu().numpy()

        # Save reconstructed image to the script directory
        try:
            recon_img = (np.clip(recon_np, 0, 1) * 255.0).round().astype(np.uint8)
            recon_pil = Image.fromarray(recon_img)
            img_out_path = os.path.join(script_dir, f"{safe_name}_recon.png")
            recon_pil.save(img_out_path)
        except Exception as e:
            print(f"  Warning: could not save reconstructed image for {model_name}: {e}")

        reconstructions.append((model_name, recon_np))
        
        print(
            f"  PSNR: {psnr.item():.2f} dB, Compression Ratio: {compression_ratio:.2f}x, "
            f"BPP: {results[-1]['bpp']:.4f}, Size: {results[-1]['compressed_kb']:.2f} KB ({results[-1]['compressed_bytes']} bytes)"
        )
        
    except Exception as e:
        print(f"  Error with {model_name}: {e}")

# Print summary
print("\n=== Summary ===")
for result in sorted(results, key=lambda x: x["psnr"], reverse=True):
    print(
        f"{result['name']}: PSNR={result['psnr']:.2f} dB, CR={result['compression_ratio']:.2f}x, "
        f"BPP={result['bpp']:.4f}, Size={result['compressed_kb']:.2f} KB ({result['compressed_bytes']} bytes)"
    )

# Visualization
# Move original back to CPU for visualization
orig = x.detach().cpu().squeeze().permute(1, 2, 0).numpy()
n_models = len(reconstructions) + 1  # +1 for original

fig, axes = plt.subplots(1, n_models, figsize=(4*n_models, 4))

# Show original
axes[0].imshow(orig)
axes[0].set_title("Original")
axes[0].axis("off")

# Show reconstructions
for idx, (model_name, recon_img) in enumerate(reconstructions):
    psnr_val = results[idx]["psnr"]
    axes[idx+1].imshow(recon_img)
    axes[idx+1].set_title(f"{model_name}\nPSNR: {psnr_val:.2f} dB")
    axes[idx+1].axis("off")

plt.tight_layout()
plt.show()
