"""Train a 2D neural image field and run hyperparameter ablation.

Usage:
    python scripts/train_2d_field.py --image photo.png
    python scripts/train_2d_field.py --image photo.png --grid_search
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader, RandomSampler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.image_mlp import ImageMLP, PixelDataset, reconstruct_image


def train_single(image_path, L=10, hidden_dim=256, lr=1e-2, num_iters=1000,
                 batch_size=10000, save_dir="results/2d_field",
                 checkpoint_iters=None):
    """Train a single 2D neural field."""
    if checkpoint_iters is None:
        checkpoint_iters = [1, 50, 100, 200, 500, 1000]

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PixelDataset(image_path)
    sampler = RandomSampler(dataset, replacement=True, num_samples=num_iters * batch_size)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    model = ImageMLP(L=L, hidden_dim=hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: L={L}, hidden_dim={hidden_dim}, {n_params:,} params")

    psnr_history = []
    model.train()
    for it, (coords, colors) in enumerate(loader, start=1):
        coords, colors = coords.to(device), colors.to(device)
        pred = model(coords)
        loss = nn.MSELoss()(pred, colors)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mse = loss.item()
        psnr = -10.0 * np.log10(mse) if mse > 0 else float("inf")
        psnr_history.append(psnr)

        if it % 100 == 0 or it == 1:
            print(f"  [{it:4d}/{num_iters}] MSE={mse:.6f}  PSNR={psnr:.1f} dB")

        if it in checkpoint_iters or it == num_iters:
            img = reconstruct_image(model, dataset, device)
            model.train()
            plt.figure(figsize=(6, 6))
            plt.imshow(img)
            plt.axis("off")
            plt.title(f"Iter {it} | PSNR: {psnr:.1f} dB")
            plt.tight_layout()
            plt.savefig(save_dir / f"iter_{it:04d}.png", bbox_inches="tight", dpi=150)
            plt.close()

    torch.save(model.state_dict(), save_dir / "model.pth")
    return model, psnr_history


def grid_search(image_path, save_dir="results/2d_field", num_iters=1000):
    """Run hyperparameter ablation: vary PE frequencies and network width."""
    save_dir = Path(save_dir)
    configs = [
        (2, 64, "Low freq, narrow"),
        (2, 256, "Low freq, wide"),
        (10, 64, "High freq, narrow"),
        (10, 256, "High freq, wide"),
    ]

    results = []
    for L, dim, label in configs:
        print(f"\n{'='*50}")
        print(f"Config: L={L}, hidden_dim={dim} ({label})")
        cfg_dir = save_dir / f"L{L}_dim{dim}"
        _, psnr_hist = train_single(
            image_path, L=L, hidden_dim=dim, num_iters=num_iters,
            save_dir=str(cfg_dir), checkpoint_iters=[num_iters],
        )
        results.append({"L": L, "dim": dim, "label": label,
                        "psnr": psnr_hist[-1], "dir": cfg_dir})

    # Create comparison grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    for ax, r in zip(axes.flatten(), results):
        img_path = r["dir"] / f"iter_{num_iters:04d}.png"
        if img_path.exists():
            ax.imshow(plt.imread(str(img_path)))
        ax.axis("off")
        ax.set_title(f"L={r['L']}, Width={r['dim']}\nPSNR: {r['psnr']:.1f} dB", fontsize=12)

    plt.suptitle("Hyperparameter Ablation", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_dir / "ablation_grid.png", dpi=150)
    plt.close()
    print(f"\nAblation grid saved to {save_dir / 'ablation_grid.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 2D neural image field")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="results/2d_field")
    parser.add_argument("--L", type=int, default=10)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--num_iters", type=int, default=1000)
    parser.add_argument("--grid_search", action="store_true")
    args = parser.parse_args()

    if args.grid_search:
        grid_search(args.image, args.save_dir, args.num_iters)
    else:
        train_single(args.image, args.L, args.hidden_dim, args.lr,
                      args.num_iters, save_dir=args.save_dir)
