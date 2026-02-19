"""Dataset loading and ray sampling for NeRF training.

- load_nerf_dataset: Load pre-processed .npz dataset files.
- RaysData: Efficient random ray sampling across multiple views for training.
"""

import torch
import numpy as np
from typing import Tuple

from .rays import pixel_to_ray


def load_nerf_dataset(
    path: str, split: str = "train"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a NeRF dataset from .npz file.

        Args:
        path: Path to the .npz file (containing images_train, images_val, c2ws_train, c2ws_val, c2ws_test, focal)
        split: 'train', 'val', 'test', 'train+val'.

    Returns:
        For train/val: (images, c2ws, K) where images are float32 in [0, 1].
        For test: (c2ws, K).
    """
    data = np.load(path)

    images_train = data["images_train"] / 255.0
    images_val = data["images_val"] / 255.0
    c2ws_train = data["c2ws_train"]
    c2ws_val = data["c2ws_val"]
    c2ws_test = data["c2ws_test"]
    focal = float(data["focal"])

    H, W = images_train.shape[1:3]
    K = np.array([[focal, 0, W / 2], [0, focal, H / 2], [0, 0, 1]])

    if split == "train":
        return images_train, c2ws_train, K
    elif split == "val":
        return images_val, c2ws_val, K
    elif split == "test":
        return c2ws_test, K
    elif split == "train+val":
        images = np.concatenate([images_train, images_val], axis=0)
        c2ws = np.concatenate([c2ws_train, c2ws_val], axis=0)
        return images, c2ws, K
    else:
        raise ValueError(f"Unknown split '{split}'. Use train/val/test/train+val.")


class RaysData:
    """Random ray sampler for NeRF training.
    Pre-computes a pixel coordinate grid and samples random rays across
    all training views during each training iteration.

    Args:
        images: Training images (N, H, W, 3) as numpy or tensor.
        K: Intrinsic matrix (3, 3).
        c2ws: Camera-to-world matrices (N, 4, 4).
    """

    def __init__(self, images, K, c2ws):
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images).float()
        if isinstance(K, np.ndarray):
            K = torch.from_numpy(K).float()
        if isinstance(c2ws, np.ndarray):
            c2ws = torch.from_numpy(c2ws).float()

        self.images = images
        self.n_images, self.H, self.W = images.shape[:3]
        self.K = K
        self.c2ws = c2ws

        # Pre-compute pixel center grid: each entry is [u + 0.5, v + 0.5]
        i, j = torch.meshgrid(
            torch.arange(self.H, dtype=torch.float32),
            torch.arange(self.W, dtype=torch.float32),
            indexing="ij",
        )
        self.uv_grid = torch.stack([j + 0.5, i + 0.5], dim=-1)  # (H, W, 2)

    def sample_rays(
        self, n_rays: int, image_idx: int | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample random rays with their ground-truth RGB values.

        Args:
            n_rays: Number of rays to sample.
            image_idx: If given, sample only from this image.

        Returns:
            rays_o: Ray origins (n_rays, 3).
            rays_d: Normalized ray directions (n_rays, 3).
            pixels: Ground-truth RGB (n_rays, 3).
        """
        if image_idx is not None:
            image_ids = torch.full((n_rays,), image_idx, dtype=torch.long)
        else:
            image_ids = torch.randint(0, self.n_images, (n_rays,))

        rows = torch.randint(0, self.H, (n_rays,))
        cols = torch.randint(0, self.W, (n_rays,))
        uv = self.uv_grid[rows, cols]
        pixels = self.images[image_ids, rows, cols]

        c2ws = self.c2ws[image_ids]
        rays_o, rays_d = pixel_to_ray(self.K, c2ws, uv)

        return rays_o, rays_d, pixels

    def get_all_rays_for_image(
        self, img_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get rays for every pixel in a single image (for rendering).

        Args:
            img_idx: Index of the image.

        Returns:
            rays_o: Ray origins (H*W, 3).
            rays_d: Ray directions (H*W, 3).
            target_rgb: Ground-truth colors (H*W, 3).
        """
        uv = self.uv_grid.reshape(-1, 2)
        c2w = self.c2ws[img_idx]
        ray_o, ray_d = pixel_to_ray(self.K, c2w, uv)
        ray_o = ray_o.unsqueeze(0).expand(self.H * self.W, -1)
        target_rgb = self.images[img_idx].reshape(-1, 3)
        return ray_o, ray_d, target_rgb
