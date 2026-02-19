"""Volume rendering for Neural Radiance Fields.

Implements the classical volume rendering equation from NeRF:
    C(r) = Σ T_i · α_i · c_i

where T_i is the accumulated transmittance and α_i = 1 - exp(-σ_i · δ_i).

Also handles point sampling along rays with stratified sampling.
"""

import torch
import numpy as np
from typing import Tuple

from .nerf_model import NeRF_MLP


def vol_rendering(
    pts_rgb: torch.Tensor,
    pts_density: torch.Tensor,
    t_vals: torch.Tensor,
    ray_d: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply volume rendering to per-sample colors and densities.

    Args:
        pts_rgb: Sample colors (N_rays, N_samples, 3).
        pts_density: Sample densities (N_rays, N_samples, 1).
        t_vals: Sample distances along rays (N_rays, N_samples).
        ray_d: Ray directions (N_rays, 3), used for distance scaling.

    Returns:
        rays_rgb: Rendered pixel colors (N_rays, 3).
        rays_depth: Expected depth per ray (N_rays,).
        weights: Per-sample rendering weights (N_rays, N_samples).
    """
    if t_vals.dim() == 1:
        t_vals = t_vals.unsqueeze(0)

    # Inter-sample distances; last sample gets large distance (background)
    dists = t_vals[:, 1:] - t_vals[:, :-1]
    dists = torch.cat(
        [dists, torch.full((dists.shape[0], 1), 1e10, device=dists.device)], dim=-1
    )

    # Alpha compositing: α_i = 1 - exp(-σ_i * δ_i)
    density = pts_density.squeeze(-1)
    alpha = 1.0 - torch.exp(-density * dists)

    # Transmittance: T_i = Π_{j<i} (1 - α_j)
    transmittance = torch.cumprod(
        torch.cat([torch.ones_like(alpha[:, :1]), 1.0 - alpha + 1e-10], dim=-1),
        dim=-1,
    )[:, :-1]

    # Rendering weights: w_i = T_i * α_i
    weights = transmittance * alpha

    # Weighted sum for final color and depth
    rays_rgb = torch.sum(weights.unsqueeze(-1) * pts_rgb, dim=1)
    rays_depth = torch.sum(weights * t_vals, dim=1)

    # White background compositing
    acc_alpha = torch.sum(weights, dim=1, keepdim=True)
    rays_rgb = rays_rgb + (1.0 - acc_alpha)

    return rays_rgb, rays_depth, weights


def render_rays(
    model: NeRF_MLP,
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    near: float = 2.0,
    far: float = 6.0,
    n_samples: int = 64,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample points along rays, query the NeRF, and volume-render.

    Uses stratified sampling: divides [near, far] into n_samples bins and
    draws one random sample per bin during training (uniform during eval).

    Args:
        model: NeRF_MLP model.
        rays_o: Ray origins (N_rays, 3).
        rays_d: Normalized ray directions (N_rays, 3).
        near: Near clipping distance.
        far: Far clipping distance.
        n_samples: Number of samples per ray.
        device: Torch device.

    Returns:
        rays_rgb: Rendered colors (N_rays, 3).
        rays_depth: Rendered depths (N_rays,).
    """
    rays_o = rays_o.to(device)
    rays_d = rays_d.to(device)
    N_rays = rays_o.shape[0] if rays_o.dim() > 1 else rays_d.shape[0]

    # Stratified sampling along rays
    t_vals = torch.linspace(near, far, n_samples, device=device)
    if model.training:
        bin_width = (far - near) / n_samples
        t_vals = t_vals.unsqueeze(0) + torch.rand(N_rays, n_samples, device=device) * bin_width
    else:
        t_vals = t_vals.unsqueeze(0).expand(N_rays, -1)

    if rays_o.dim() == 1:
        rays_o = rays_o.unsqueeze(0).expand(N_rays, -1)

    # Compute 3D sample positions: p = o + t * d
    pts = rays_o.unsqueeze(1) + t_vals.unsqueeze(-1) * rays_d.unsqueeze(1)
    dirs = rays_d.unsqueeze(1).expand(-1, n_samples, -1)

    # Query NeRF
    pts_flat = pts.reshape(-1, 3)
    dirs_flat = dirs.reshape(-1, 3)
    pts_rgb, pts_density = model(pts_flat, dirs_flat)
    pts_rgb = pts_rgb.reshape(N_rays, n_samples, 3)
    pts_density = pts_density.reshape(N_rays, n_samples, 1)

    # Volume render
    rays_rgb, rays_depth, _ = vol_rendering(pts_rgb, pts_density, t_vals, rays_d)
    return rays_rgb, rays_depth


def render_full_image(
    model: NeRF_MLP,
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    chunk_size: int = 1024,
    near: float = 2.0,
    far: float = 6.0,
    device: str = "cuda",
) -> torch.Tensor:
    """Render all rays for a complete image, processing in chunks.

    Args:
        model: NeRF_MLP model.
        rays_o: All ray origins (H*W, 3).
        rays_d: All ray directions (H*W, 3).
        chunk_size: Rays per forward pass (limits GPU memory usage).
        near: Near plane.
        far: Far plane.
        device: Torch device.

    Returns:
        Rendered RGB values (H*W, 3) on CPU.
    """
    rays_o = rays_o.to(device)
    rays_d = rays_d.to(device)
    model.eval()

    with torch.no_grad():
        chunks = []
        for i in range(0, len(rays_o), chunk_size):
            rgb, _ = render_rays(
                model,
                rays_o[i : i + chunk_size],
                rays_d[i : i + chunk_size],
                near=near,
                far=far,
                device=device,
            )
            chunks.append(rgb.cpu())

    return torch.cat(chunks)
