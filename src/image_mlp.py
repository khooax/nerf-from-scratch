"""2D Neural Image Field 

- map 2D pixel coords (x, y) → RGB color (r, g, b) using sinusoidal PE and a 4-layer MLP; 
- simpler precursor to full 3D NERF
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .positional_encoding import SinusoidalPE


class ImageMLP(nn.Module):
    """maps encoded 2D coords to RGB colors

    Architecture:
        PE(x,y) → Linear(256) → ReLU → Linear(256) → ReLU →
        Linear(256) → ReLU → Linear(3) → Sigmoid

    Args:
        L: # positional encoding frequency levels
        hidden_dim: Width of hidden layers
    """

    def __init__(self, L: int = 10, hidden_dim: int = 256):
        super().__init__()
        self.pe = SinusoidalPE(L=L, input_dim=2)
        input_dim = self.pe.output_dim

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid(),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """predict RGB for given pixel coords

        Args:
            coords: Normalized pixel coords (N, 2) in [0, 1]

        Returns:
            RGB colors (N, 3) in [0, 1].
        """
        return self.layers(self.pe(coords))


class PixelDataset(Dataset):
    """dataset of (coordinate, color) pairs from an image, 
    normalizes pixel coords to [0, 1] and RGB values to [0, 1]

    Args:
        image_path: Path to the input image.
    """

    def __init__(self, image_path: str):
        img = Image.open(image_path).convert("RGB")
        self.image = np.array(img).astype(np.float32) / 255.0
        self.H, self.W = self.image.shape[:2]

        y_coords, x_coords = np.meshgrid(
            np.arange(self.H), np.arange(self.W), indexing="ij"
        )
        self.coords = np.stack(
            [x_coords.flatten() / self.W, y_coords.flatten() / self.H], axis=-1
        ).astype(np.float32)
        self.colors = self.image.reshape(-1, 3)

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        return self.coords[idx], self.colors[idx]


def reconstruct_image(
    model: ImageMLP, dataset: PixelDataset, device: torch.device, batch_size: int = 100000
) -> np.ndarray:
    """reconstruct full image by querying  model at every pixel

    Args:
        dataset: PixelDataset with coordinate grid
        device: Torch device
        batch_size: # pixels to process at once

    Returns:
        Reconstructed image as (H, W, 3) numpy array in [0, 1].
    """
    model.eval()
    coords = torch.from_numpy(dataset.coords).to(device)

    with torch.no_grad():
        predictions = []
        for i in range(0, len(coords), batch_size):
            pred = model(coords[i : i + batch_size])
            predictions.append(pred.cpu())
        predictions = torch.cat(predictions, dim=0)

    img = predictions.numpy().reshape(dataset.H, dataset.W, 3)
    return np.clip(img, 0, 1)
