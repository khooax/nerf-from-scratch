"""Ray generation from camera parameters
- pixel (u, v) → camera coords → world coords → ray (origin, direction)
"""

import torch
from typing import Tuple

def transform_points(c2w: torch.Tensor, xc: torch.Tensor) -> torch.Tensor:
    """transform 3D points from camera to world coords

    Args:
        c2w: Camera-to-world matrix, (4, 4) or (N, 4, 4)
        xc: Points in camera space, (N, 3)
    Returns:
        Points in world space, (N, 3)
    """
    xch = torch.cat([xc, torch.ones_like(xc[..., :1])], dim=-1)

    if c2w.dim() == 2:
        xwh = xch @ c2w.T
    else:
        xwh = torch.bmm(xch.unsqueeze(1), c2w.transpose(1, 2)).squeeze(1)

    return xwh[..., :3]


def pixel_to_camera(
    K: torch.Tensor, uv: torch.Tensor, s: torch.Tensor | float
) -> torch.Tensor:
    """convert pixels to camera-space 3D points at a given depth w inverse pinhole camera model
        x = (u - cx) / fx * s
        y = (v - cy) / fy * s
        z = s

    Args:
        K: Intrinsic matrix (3, 3)
        uv: Pixel coords (N, 2)
        s: Depth values — scalar, (N,), or (N, 1)
    Returns:
        Camera-space 3D points (N, 3)
    """
    if isinstance(s, (int, float)):
        s = torch.ones(uv.shape[0], 1) * s
    elif s.dim() == 1:
        s = s.unsqueeze(-1)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u, v = uv[:, 0:1], uv[:, 1:2]

    x = (u - cx) / fx * s
    y = (v - cy) / fy * s
    z = s

    return torch.cat([x, y, z], dim=-1)


def pixel_to_ray(
    K: torch.Tensor, c2w: torch.Tensor, uv: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """convert pixel coords to rays by projecting each pixel to depth=1 in camera space
    transforming to world space, then computing the normalized direction from camera origin

    Args:
        K: Intrinsic matrix (3, 3)
        c2w: Camera-to-world matrix, (4, 4) or (N, 4, 4) for batched cameras
        uv: Pixel coords (N, 2)

    Returns:
        ray_o: Ray origins — (3,) for single camera, (N, 3) for batched
        ray_d: Normalized ray directions (N, 3)
    """
    xc = pixel_to_camera(K, uv, s=1.0)
    xw = transform_points(c2w, xc)

    if c2w.dim() == 2:
        ray_o = c2w[:3, 3]
        ray_d = xw - ray_o
    else:
        ray_o = c2w[..., :3, 3]
        ray_d = xw - ray_o

    ray_d = ray_d / torch.norm(ray_d, dim=-1, keepdim=True)
    return ray_o, ray_d
