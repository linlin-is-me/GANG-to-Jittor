"""Jittor replacements for kornia functions used in GANG.

Replaces:
- kornia.morphology.erosion → jt_erosion2d
- kornia.filters.spatial_gradient → jt_spatial_gradient
"""

import jittor as jt
import jittor.nn as F


def jt_erosion2d(x: jt.Var, kernel_size: int = 7) -> jt.Var:
    """Binary erosion on 2D mask tensor.

    Equivalent to: kornia.morphology.erosion(mask[None, ...], kernel)[0]

    Erosion is the morphological operation that shrinks bright regions.
    For a binary (0/1) mask: erosion(x) = min over kernel neighborhood.
    Can be implemented as: -(max_pool2d(-x)) = -dilation(-x)

    Args:
        x: Input tensor [1, 1, H, W] (binary mask, float)
        kernel_size: Size of the square structuring element
    Returns:
        Eroded tensor [1, 1, H, W]
    """
    padding = kernel_size // 2
    neg_mask = -(x.float())
    eroded_neg = F.max_pool2d(neg_mask, kernel_size, stride=1, padding=padding)
    return -eroded_neg


def jt_spatial_gradient(x: jt.Var, order: int = 1) -> jt.Var:
    """Compute spatial gradients of a tensor.

    Equivalent to: kornia.filters.spatial_gradient(data, order=1)

    Returns gradients in x and y directions using finite differences.
    Output shape: [B, C, 2, H, W] where [..., 0, :, :] is dy and [..., 1, :, :] is dx.

    Args:
        x: Input tensor [B, C, H, W]
        order: Gradient order (only order=1 is supported)
    Returns:
        Gradients tensor [B, C, 2, H, W]
    """
    if order != 1:
        raise NotImplementedError("Only order=1 supported")

    B, C, H, W = x.shape

    # Gradient in x direction (along width): dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dx = F.pad(dx, (1, 0, 0, 0), mode='constant', value=0.0)  # pad left

    # Gradient in y direction (along height): dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    dy = F.pad(dy, (0, 0, 1, 0), mode='constant', value=0.0)  # pad top

    # Stack: [B, C, 2, H, W] where dim=2 index 0=dy, 1=dx
    return jt.stack([dy, dx], dim=2)
