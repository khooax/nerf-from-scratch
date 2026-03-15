"""NERF MLP architecture"""

import torch
import torch.nn as nn

from .positional_encoding import SinusoidalPE


class NeRF_MLP(nn.Module):
    """ 
    Architecture follows NeRF paper
    - 8-layer MLP for coordinate processing with a skip connection at layer 5
    - separate density head (position-only) and color head (position + view direction)
    - sinusoidal PE for both coords and directions

    Args:
        L_coord: Positional encoding levels for 3D coords
        L_dir: Positional encoding levels for view directions
        hidden_dim: Hidden layer width
    """

    def __init__(self, L_coord: int = 10, L_dir: int = 4, hidden_dim: int = 256):
        super().__init__()

        self.coord_PE = SinusoidalPE(L=L_coord, input_dim=3)
        self.dir_PE = SinusoidalPE(L=L_dir, input_dim=3)

        coord_dim = self.coord_PE.output_dim  # 3 + 2*3*10 = 63
        dir_dim = self.dir_PE.output_dim  # 3 + 2*3*4 = 27

        # First 4 layers: process coords
        self.coord_layers_1 = nn.Sequential(
            nn.Linear(coord_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # next 4 layers: skip connection (concat original PE with features)
        self.coord_layers_2 = nn.Sequential(
            nn.Linear(coord_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # density: depends only on position (view-independent)
        self.density_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.ReLU(),  # σ ≥ 0
        )

        # color: depends on position features + view direction
        self.color_head = nn.Sequential(
            nn.Linear(dir_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),
            nn.Sigmoid(),  # RGB ∈ [0, 1]
        )

    def forward(
        self, coords: torch.Tensor, dirs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """query radiance field at given 3D points and view directions.

        Args:
            coords: 3D sample positions (N, 3).
            dirs: Viewing directions (N, 3), should be normalized.

        Returns:
            rgb: Predicted colors (N, 3) in [0, 1].
            density: Volume density (N, 1), non-negative.
        """
        coords_enc = self.coord_PE(coords)
        dirs_enc = self.dir_PE(dirs)

        # Coordinate processing with skip connection
        h = self.coord_layers_1(coords_enc)
        h = self.coord_layers_2(torch.cat([h, coords_enc], dim=-1))

        density = self.density_head(h)
        rgb = self.color_head(torch.cat([h, dirs_enc], dim=-1))

        return rgb, density
