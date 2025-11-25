"""
CompressAI Model Comparison Script

Loads a random image from the landscapes dataset and applies 5 compression models.
Displays original + 5 reconstructions with PSNR and BPP metrics.
"""

import os
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from compressai.zoo import (
    bmshj2018_factorized,
    bmshj2018_hyperprior,
    cheng2020_anchor,
    mbt2018,
    mbt2018_mean,
)


def compute_psnr(original, reconstruction):
    """Compute PSNR between two tensors."""
    mse = torch.mean((original - reconstruction).pow(2))
    if mse == 0:
        return float('inf')
    return 10 * torch.log10(1 / mse)


def compute_bpp(strings, height, width):
    """Compute bits per pixel from compressed strings."""
    total_bytes = 0
    for s in strings:
        if isinstance(s, (bytes, bytearray)):
            total_bytes += len(s)
        elif isinstance(s, (list, tuple)):
            for item in s:
                if isinstance(item, (bytes, bytearray)):
                    total_bytes += len(item)
    total_bits = total_bytes * 8
    num_pixels = height * width
    return total_bits / num_pixels if num_pixels > 0 else 0


def load_landscape_image():
    """Load a random image from images/landscapes/test folder."""
    script_dir = Path(__file__).parent
    landscapes_dir = script_dir / "images" / "landscapes" / "test"
    
    if not landscapes_dir.exists():
        raise FileNotFoundError(f"Landscapes directory not found: {landscapes_dir}")
    
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    image_files = [f for f in landscapes_dir.iterdir() if f.suffix.lower() in image_extensions]
    
    if not image_files:
        raise FileNotFoundError(f"No images found in {landscapes_dir}")
    
    image_path = random.choice(image_files)
    print(f"Loading image: {image_path.name}")
    
    img = Image.open(image_path).convert("RGB")
    
    # Resize to dimensions divisible by 64 (required by models)
    w, h = img.size
    w = (w // 64) * 64
    h = (h // 64) * 64
    img = img.resize((w, h), Image.Resampling.LANCZOS)
    
    return img


def main():
    # Get quality level from user
    while True:
        try:
            quality = int(input("Enter desired quality level (1-6): "))
            if 1 <= quality <= 6:
                break
            print("Please enter a number between 1 and 6.")
        except ValueError:
            print("Please enter a valid number.")
    
    print(f"\nUsing quality level: {quality}\n")
    
    # Load image
    img = load_landscape_image()
    
    # Convert to tensor [1, 3, H, W], normalized to [0,1]
    x = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    x = x.to(device)
    
    #checkpoint paths
    factorized_ckpt = "checkpoint_best_loss_factorized1e-3.pth.tar"  
    hyperprior_ckpt = "checkpoint_best_loss_hyperprior1e-3.pth.tar"  

    # Prompt user for pretrained or retrained for each model
    def ask_choice(model_name):
        while True:
            choice = input(f"Use retrained checkpoint for {model_name}? (y/n): ").strip().lower()
            if choice in ("y", "n"): return choice == "y"
            print("Please enter 'y' or 'n'.")

    use_factorized_retrained = ask_choice("bmshj2018-factorized")
    use_hyperprior_retrained = ask_choice("bmshj2018-hyperprior")

    models = []
    # Factorized Prior
    if use_factorized_retrained:
        print(f"Loading retrained checkpoint for bmshj2018-factorized: {factorized_ckpt}")
        model = bmshj2018_factorized(quality=quality, pretrained=False)
        checkpoint = torch.load(factorized_ckpt, map_location=device)

        model.load_state_dict(checkpoint)
        model = model.eval().to(device)
        models.append(('Factorized Prior (Retrained, 10 Epochs)', model))
    else:
        models.append(('Factorized Prior (Pretrained)', bmshj2018_factorized(quality=quality, pretrained=True)))

    # Scale Hyperprior
    if use_hyperprior_retrained:
        print(f"Loading retrained checkpoint for bmshj2018-hyperprior: {hyperprior_ckpt}")
        model = bmshj2018_hyperprior(quality=quality, pretrained=False)
        checkpoint = torch.load(hyperprior_ckpt, map_location=device)
       
        model.load_state_dict(checkpoint)
        model = model.eval().to(device)
        models.append(('Scale Hyperprior (Retrained, 10 Epochs)', model))
    else:
        models.append(('Scale Hyperprior (Pretrained)', bmshj2018_hyperprior(quality=quality, pretrained=True)))

    # The rest always use pretrained
    models += [
        ('Mean-Scale Hyperprior', mbt2018_mean(quality=quality, pretrained=True)),
        ('Autoregressive', mbt2018(quality=quality, pretrained=True)),
        ('Cheng2020 Anchor', cheng2020_anchor(quality=quality, pretrained=True)),
    ]
    
    results = []
    
    # Process each model
    for model_name, model in models:
        print(f"Processing {model_name}...")
        model = model.eval().to(device)
        
        with torch.no_grad():
            # Compress with timing
            start_time = time.time()
            compressed = model.compress(x)
            encode_time = time.time() - start_time
            
            # Decompress with timing
            start_time = time.time()
            reconstructed = model.decompress(compressed["strings"], compressed["shape"])
            decode_time = time.time() - start_time
            
            x_hat = reconstructed["x_hat"].clamp(0, 1)
            
            # Compute metrics
            psnr = compute_psnr(x, x_hat).item()
            bpp = compute_bpp(compressed["strings"], x.shape[2], x.shape[3])
            
            # Store results
            recon_np = x_hat.squeeze().permute(1, 2, 0).cpu().numpy()
            results.append({
                'name': model_name,
                'reconstruction': recon_np,
                'psnr': psnr,
                'bpp': bpp,
                'encode_time': encode_time,
                'decode_time': decode_time
            })
            
            print(f"  PSNR: {psnr:.2f} dB, BPP: {bpp:.4f}")
            print(f"  Encode: {encode_time*1000:.1f} ms, Decode: {decode_time*1000:.1f} ms\n")
    
    # Display results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Show original
    orig_np = x.squeeze().permute(1, 2, 0).cpu().numpy()
    axes[0].imshow(orig_np)
    axes[0].set_title("Original", fontsize=12, fontweight='bold')
    axes[0].axis("off")
    
    # Show reconstructions
    for idx, result in enumerate(results):
        axes[idx + 1].imshow(result['reconstruction'])
        axes[idx + 1].set_title(
            f"{result['name']}\n"
            f"PSNR: {result['psnr']:.2f} dB | BPP: {result['bpp']:.4f}\n"
            f"Enc: {result['encode_time']*1000:.1f}ms | Dec: {result['decode_time']*1000:.1f}ms",
            fontsize=9
        )
        axes[idx + 1].axis("off")
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
