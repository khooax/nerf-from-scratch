"""Training script for NeRF.

Usage:
    python scripts/train_nerf.py --config configs/lego.yaml
    python scripts/train_nerf.py --dataset lego_200x200.npz --near 2.0 --far 6.0
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.nerf_model import NeRF_MLP
from src.dataset import RaysData, load_nerf_dataset
from src.rendering import render_rays, render_full_image


def train(args):
    device = torch.device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    images_train, c2ws_train, K = load_nerf_dataset(args.dataset, "train")
    images_val, c2ws_val, _ = load_nerf_dataset(args.dataset, "val")
    train_data = RaysData(images_train, K, c2ws_train)
    val_data = RaysData(images_val, K, c2ws_val)

    # Model
    model = NeRF_MLP(
        L_coord=args.L_coord, L_dir=args.L_dir, hidden_dim=args.hidden_dim
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    mse_loss = nn.MSELoss()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters | Device: {device}")
    print(f"Training: {args.num_iters} iters, batch_size={args.batch_size}, lr={args.lr}")
    print(f"Rendering: near={args.near}, far={args.far}")

    # Training loop
    train_losses = []
    val_psnrs = []
    checkpoint_iters = set([200, 400, 600, 800, 1000] + [args.num_iters])

    for it in range(1, args.num_iters + 1):
        model.train()

        rays_o, rays_d, target = train_data.sample_rays(args.batch_size)
        pred, _ = render_rays(
            model, rays_o, rays_d, near=args.near, far=args.far, device=device
        )

        loss = mse_loss(pred, target.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        if it % 50 == 0 or it == 1:
            psnr = -10.0 * np.log10(loss.item())
            print(f"[{it:5d}/{args.num_iters}] loss={loss.item():.6f}  PSNR={psnr:.1f} dB")

        # Validation + checkpoint
        if it in checkpoint_iters:
            model.eval()
            val_psnr = evaluate(model, val_data, args.near, args.far, device)
            val_psnrs.append((it, val_psnr))
            print(f"  → Validation PSNR: {val_psnr:.2f} dB")

            save_render(model, val_data, 0, save_dir / f"val_{it:05d}.png",
                        args.near, args.far, device)

    # Save model and plots
    torch.save(model.state_dict(), save_dir / "model.pth")
    plot_curves(train_losses, val_psnrs, save_dir / "training_curves.png")
    print(f"\nDone. Results saved to {save_dir}/")


def evaluate(model, dataset, near, far, device):
    """Compute mean PSNR across all validation images."""
    psnrs = []
    for idx in range(dataset.n_images):
        rays_o, rays_d, target = dataset.get_all_rays_for_image(idx)
        pred = render_full_image(model, rays_o, rays_d, near=near, far=far, device=device)
        mse = torch.mean((pred - target) ** 2)
        psnrs.append(-10.0 * torch.log10(mse).item())
    return np.mean(psnrs)


def save_render(model, dataset, idx, path, near, far, device):
    """Render and save a single image."""
    rays_o, rays_d, _ = dataset.get_all_rays_for_image(idx)
    pred = render_full_image(model, rays_o, rays_d, near=near, far=far, device=device)
    img = np.clip(pred.numpy().reshape(dataset.H, dataset.W, 3), 0, 1)
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()


def plot_curves(losses, val_psnrs, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(losses)
    ax1.set(xlabel="Iteration", ylabel="MSE Loss", title="Training Loss")
    ax1.grid(True, alpha=0.3)

    if val_psnrs:
        iters, psnrs = zip(*val_psnrs)
        ax2.plot(iters, psnrs, marker="o")
        ax2.set(xlabel="Iteration", ylabel="PSNR (dB)", title="Validation PSNR")
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NeRF")
    parser.add_argument("--dataset", type=str, required=True, help="Path to .npz dataset")
    parser.add_argument("--save_dir", type=str, default="results/nerf")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_iters", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--L_coord", type=int, default=10)
    parser.add_argument("--L_dir", type=int, default=4)
    parser.add_argument("--near", type=float, default=2.0)
    parser.add_argument("--far", type=float, default=6.0)
    train(parser.parse_args())
