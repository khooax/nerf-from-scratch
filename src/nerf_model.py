"""Neural Radiance Field MLP architecture.

Implements the NeRF model from Mildenhall et al. (2020) with:
- 8-layer MLP for coordinate processing with a skip connection at layer 5
- Separate density head (position-only) and color head (position + view direction)
- Sinusoidal positional encoding for both coordinates and directions
"""

import torch
import torch.nn as nn

from .positional_encoding import SinusoidalPE


class NeRF_MLP(nn.Module):
    """NeRF MLP with skip connections and view-dependent color prediction.

    Architecture (following the original NeRF paper):
        coords → PE → 4×(Linear+ReLU) → skip connection → 4×(Linear+ReLU)
            ├── density head → σ (ReLU, non-negative)
            └── + dir PE → 2×(Linear+ReLU) → Linear → RGB (Sigmoid)

    Args:
        L_coord: Positional encoding levels for 3D coordinates.
        L_dir: Positional encoding levels for view directions.
        hidden_dim: Hidden layer width.
    """

    def __init__(self, L_coord: int = 10, L_dir: int = 4, hidden_dim: int = 256):
        super().__init__()

        self.coord_PE = SinusoidalPE(L=L_coord, input_dim=3)
        self.dir_PE = SinusoidalPE(L=L_dir, input_dim=3)

        coord_dim = self.coord_PE.output_dim  # 3 + 2*3*10 = 63
        dir_dim = self.dir_PE.output_dim  # 3 + 2*3*4 = 27

        # First 4 layers: process coordinates
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

        # Next 4 layers: skip connection (concat original PE with features)
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

        # Density: depends only on position (view-independent)
        self.density_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.ReLU(),  # σ ≥ 0
        )

        # Color: depends on position features + view direction
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
        """Query the radiance field at given 3D points and view directions.

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
