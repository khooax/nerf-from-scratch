"""Sinusoidal PE 
- lifts low-dim inputs into a higher-dim space allowing MLPs to learn high-freq funcs
"""

import torch
import numpy as np


class SinusoidalPE:
    """
    Args:
        L: #frequency levels
        input_dim: Dimension of input coords (2 for images, 3 for 3D points)

    Output dims: input_dim + 2 * input_dim * L
        (original coords + sin/cos at each frequency for each dimension)
    """

    def __init__(self, L: int, input_dim: int = 3):
        self.L = L
        self.input_dim = input_dim
        self.output_dim = input_dim + 2 * input_dim * L

    def __call__(self, coords: torch.Tensor) -> torch.Tensor: 
        out = [coords]
        for l in range(self.L):
            freq = (2.0 ** l) * np.pi
            out.append(torch.sin(coords * freq))
            out.append(torch.cos(coords * freq))
        return torch.cat(out, dim=-1)
