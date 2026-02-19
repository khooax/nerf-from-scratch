"""Sinusoidal positional encoding for neural fields.
    γ(p) = [p, sin(2^0 π p), cos(2^0 π p), ..., sin(2^(L-1) π p), cos(2^(L-1) π p)]

This mapping lifts low-dimensional inputs into a higher-dimensional space,
allowing MLPs to learn high-frequency functions.
"""

import torch
import numpy as np


class SinusoidalPE:
    """Sinusoidal positional encoding

    Args:
        L: Number of frequency levels.
        input_dim: Dimension of input coordinates (2 for images, 3 for 3D points).

    Output dimension: input_dim + 2 * input_dim * L
        (original coords + sin/cos at each frequency for each dimension)
    """

    def __init__(self, L: int, input_dim: int = 3):
        self.L = L
        self.input_dim = input_dim
        self.output_dim = input_dim + 2 * input_dim * L

    def __call__(self, coords: torch.Tensor) -> torch.Tensor:
        """Apply positional encoding.

        Args:
            coords: Input coordinates of shape (N, input_dim).

        Returns:
            Encoded coordinates of shape (N, output_dim).
        """
        out = [coords]
        for l in range(self.L):
            freq = (2.0 ** l) * np.pi
            out.append(torch.sin(coords * freq))
            out.append(torch.cos(coords * freq))
        return torch.cat(out, dim=-1)
