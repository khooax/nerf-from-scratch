"""Render novel views from a trained NeRF model.

Usage:
    python scripts/render_novel_views.py --checkpoint results/nerf/model.pth \
        --dataset lego_200x200.npz --output novel_views.mp4
"""

import argparse
import os
import sys

import torch
import numpy as np
from tqdm import tqdm
import imageio.v2 as imageio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.nerf_model import NeRF_MLP
from src.rays import pixel_to_ray
from src.rendering import render_full_image
from src.dataset import load_nerf_dataset


def render(args):
    device = torch.device(args.device)

    # Load test poses and intrinsics
    c2ws_test, K = load_nerf_dataset(args.dataset, "test")
    K = torch.from_numpy(K).float()
    c2ws_test = torch.from_numpy(c2ws_test).float()

    # Infer image size from training set
    images_train, _, _ = load_nerf_dataset(args.dataset, "train")
    H, W = images_train.shape[1:3]
    del images_train

    # Load model
    model = NeRF_MLP(
        L_coord=args.L_coord, L_dir=args.L_dir, hidden_dim=args.hidden_dim
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # Render each test view
    frames = []
    i_grid, j_grid = torch.meshgrid(
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing="ij",
    )
    uv = torch.stack([j_grid + 0.5, i_grid + 0.5], dim=-1).reshape(-1, 2)

    print(f"Rendering {len(c2ws_test)} novel views at {H}x{W}...")
    for c2w in tqdm(c2ws_test):
        ray_o, ray_d = pixel_to_ray(K, c2w, uv)
        ray_o = ray_o.unsqueeze(0).expand(H * W, -1)

        pred = render_full_image(
            model, ray_o, ray_d, chunk_size=args.chunk_size,
            near=args.near, far=args.far, device=device,
        )
        frame = np.clip(pred.numpy().reshape(H, W, 3) * 255, 0, 255).astype(np.uint8)
        frames.append(frame)

    imageio.mimsave(args.output, frames, fps=args.fps)
    print(f"Saved {len(frames)} frames to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render novel NeRF views")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output", type=str, default="novel_views.mp4")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--near", type=float, default=2.0)
    parser.add_argument("--far", type=float, default=6.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--L_coord", type=int, default=10)
    parser.add_argument("--L_dir", type=int, default=4)
    render(parser.parse_args())
