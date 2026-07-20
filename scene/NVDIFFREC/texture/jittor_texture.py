"""
Pure Jittor texture sampling implementation (Phase 35).

Replaces nvdiffrast dr.texture() with F.grid_sample.
No CUDA compilation required — pure Jittor autograd compatible.

Supports:
- 2D textures: F.grid_sample directly
- Cubemap textures: cube face projection + F.grid_sample per face
- Cubemap mipmap: trilinear interpolation between mip levels
"""

import jittor as jt
import jittor.nn as F


def texture(tex, uv, uv_da=None, mip_level_bias=None, mip=None,
            filter_mode='auto', boundary_mode='wrap', max_mip_level=None):
    """Texture sampling — pure Jittor F.grid_sample replacement for dr.texture().

    Args:
        tex: [1, H, W, C] for 2D, [1, 6, H, W, C] for cubemap
        uv:  [B, H, W, 2] for 2D, [B, H, W, 3] for cubemap, or [N, D] flat
        uv_da: (unused, accepted for compatibility)
        mip_level_bias: per-pixel mip bias [B, H, W] or [N]
        mip: list of tensors [1, 6, H_i, W_i, C] for custom mip stack
        filter_mode: 'nearest', 'linear', 'linear-mipmap-linear', 'linear-mipmap-nearest', 'auto'
        boundary_mode: 'cube', 'clamp', 'wrap', 'zero'
        max_mip_level: (unused, accepted for compatibility)

    Returns:
        Sampled tensor [B, H, W, C] or [1, N, C]
    """
    # Default filter mode
    if filter_mode == 'auto':
        filter_mode = 'linear-mipmap-linear' if (uv_da is not None or mip_level_bias is not None) else 'linear'
    if max_mip_level == 0 and 'mipmap' in filter_mode:
        filter_mode = 'linear'

    is_cube = (boundary_mode == 'cube')
    use_mip = ('mipmap' in filter_mode)

    # ---- Dispatch ----
    if use_mip:
        return _texture_cube_mip(tex, uv, mip, mip_level_bias, filter_mode)
    elif is_cube:
        return _texture_cube(tex, uv, filter_mode)
    else:
        return _texture_2d(tex, uv, filter_mode, boundary_mode)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _texture_2d(tex, uv, filter_mode, boundary_mode):
    """Sample 2D texture [1, H, W, C] at UV [..., 2] in [0,1] range."""
    spatial_shape = uv.shape[:-1]  # [B, H, W] or [N]
    uv_flat = uv.reshape(-1, 2)  # [N, 2]

    # UV [0,1] → grid [-1,1]
    grid = (uv_flat * 2.0 - 1.0).reshape(1, uv_flat.shape[0], 1, 2)

    # tex [1, H, W, C] → [1, C, H, W]
    tex_nchw = tex.permute(0, 3, 1, 2)

    padding = {'zero': 'zeros', 'clamp': 'border', 'wrap': 'border'}.get(boundary_mode, 'zeros')
    mode = 'bilinear' if 'linear' in filter_mode else 'nearest'

    out = F.grid_sample(tex_nchw, grid, mode=mode, padding_mode=padding, align_corners=False)
    # [1, C, N, 1] → [N, C]
    result = out[0, :, :, 0].permute(1, 0)

    # Restore spatial dims
    if len(spatial_shape) >= 2:
        result = result.reshape(*(list(spatial_shape) + [-1]))
    else:
        result = result.reshape(1, spatial_shape[0], -1)
    return result


def _texture_cube(tex, uv, filter_mode):
    """Sample cubemap [1, 6, H, W, C] at 3D directions [..., 3]."""
    spatial_shape = uv.shape[:-1]
    uv_flat = uv.reshape(-1, 3)
    N = uv_flat.shape[0]
    C = tex.shape[-1]
    mode = 'bilinear' if 'linear' in filter_mode else 'nearest'

    # Normalize directions
    d = uv_flat / (uv_flat.norm(p=2, dim=-1, keepdim=True) + 1e-10)
    dx, dy, dz = d[..., 0], d[..., 1], d[..., 2]

    result = jt.zeros((N, C), dtype=tex.dtype)

    for s in range(6):
        # Project to face UV (same as _cubemap_sample_jt in util.py)
        if s == 0:   # +X
            u, v = -dz / (dx + 1e-10), -dy / (dx + 1e-10)
            in_face = (dx >= jt.abs(dy)) & (dx >= jt.abs(dz))
        elif s == 1: # -X
            u, v = dz / (-dx + 1e-10), -dy / (-dx + 1e-10)
            in_face = (-dx >= jt.abs(dy)) & (-dx >= jt.abs(dz))
        elif s == 2: # +Y
            u, v = dx / (dy + 1e-10), dz / (dy + 1e-10)
            in_face = (dy >= jt.abs(dx)) & (dy >= jt.abs(dz))
        elif s == 3: # -Y
            u, v = dx / (-dy + 1e-10), -dz / (-dy + 1e-10)
            in_face = (-dy >= jt.abs(dx)) & (-dy >= jt.abs(dz))
        elif s == 4: # +Z
            u, v = dx / (dz + 1e-10), -dy / (dz + 1e-10)
            in_face = (dz >= jt.abs(dx)) & (dz >= jt.abs(dy))
        else:        # -Z
            u, v = -dx / (-dz + 1e-10), -dy / (-dz + 1e-10)
            in_face = (-dz >= jt.abs(dx)) & (-dz >= jt.abs(dy))

        # UV [-1,1] → [0,1] → grid [-1,1]
        uv_01 = jt.stack([u * 0.5 + 0.5, v * 0.5 + 0.5], dim=-1)
        grid = (uv_01 * 2.0 - 1.0).reshape(1, N, 1, 2)

        face_tex = tex[0, s:s+1]  # [1, H, W, C]
        face_nchw = face_tex.permute(0, 3, 1, 2)
        sampled = F.grid_sample(face_nchw, grid, mode=mode, padding_mode='border', align_corners=False)
        sampled = sampled[0, :, :, 0].permute(1, 0)  # [N, C]

        mask = in_face.float().unsqueeze(-1)
        result = result + sampled * mask

    # Restore spatial dims
    if len(spatial_shape) >= 2:
        result = result.reshape(*(list(spatial_shape) + [C]))
    else:
        result = result.reshape(1, spatial_shape[0], C)
    return result


def _texture_cube_mip(tex, uv, mip, mip_level_bias, filter_mode):
    """Sample cubemap with trilinear mipmap interpolation."""
    spatial_shape = uv.shape[:-1]
    uv_flat = uv.reshape(-1, 3)
    N = uv_flat.shape[0]
    C = tex.shape[-1]

    # Build mip list
    if mip is None or not isinstance(mip, list):
        mip_list = []
    else:
        mip_list = list(mip)

    num_mips = len(mip_list)

    # Mip level bias: [B, H, W] or [B, 1, H, W] or [N] → [N]
    if mip_level_bias is None:
        lvl = jt.zeros(N)
    else:
        lvl = mip_level_bias.reshape(N)

    max_lvl = float(num_mips)
    lvl = lvl.clamp(0.0, max_lvl - 0.001)
    lvl_lo = lvl.floor().int()
    lvl_hi = (lvl_lo + 1).clamp(0, int(max_lvl))
    frac = (lvl - lvl_lo.float()).reshape(N, 1)

    # Sample base level (tex = level 0)
    c0 = _sample_single(tex, uv_flat, 'linear') if num_mips >= 0 else _sample_single(tex, uv_flat, 'linear')

    # Accumulate per-mip samples
    c_lo = jt.zeros((N, C))
    c_hi = jt.zeros((N, C))

    for li in range(num_mips):
        in_lo = (lvl_lo == (li + 1)).float().unsqueeze(-1)
        in_hi = (lvl_hi == (li + 1)).float().unsqueeze(-1)
        if in_lo.max() > 0 or in_hi.max() > 0:
            sampled = _sample_single(mip_list[li], uv_flat, 'linear')
            c_lo = c_lo + sampled * in_lo
            c_hi = c_hi + sampled * in_hi

    # Blend: level 0 uses c0, others use c_lo/c_hi
    is_lo_zero = (lvl_lo == 0).float().unsqueeze(-1)
    lo_sample = c0 * is_lo_zero + c_lo * (1.0 - is_lo_zero)
    hi_sample = c_hi
    result = lo_sample * (1.0 - frac) + hi_sample * frac

    # Restore spatial dims
    if len(spatial_shape) >= 2:
        result = result.reshape(*(list(spatial_shape) + [C]))
    else:
        result = result.reshape(1, spatial_shape[0], C)
    return result


def _sample_single(tex, uv_flat, filter_mode):
    """Sample a single cubemap level (no mip blending)."""
    N = uv_flat.shape[0]
    C = tex.shape[-1]
    d = uv_flat / (uv_flat.norm(p=2, dim=-1, keepdim=True) + 1e-10)
    dx, dy, dz = d[..., 0], d[..., 1], d[..., 2]
    result = jt.zeros((N, C), dtype=tex.dtype)
    mode = 'bilinear' if 'linear' in filter_mode else 'nearest'

    for s in range(6):
        if s == 0:
            u, v = -dz / (dx + 1e-10), -dy / (dx + 1e-10)
            in_face = (dx >= jt.abs(dy)) & (dx >= jt.abs(dz))
        elif s == 1:
            u, v = dz / (-dx + 1e-10), -dy / (-dx + 1e-10)
            in_face = (-dx >= jt.abs(dy)) & (-dx >= jt.abs(dz))
        elif s == 2:
            u, v = dx / (dy + 1e-10), dz / (dy + 1e-10)
            in_face = (dy >= jt.abs(dx)) & (dy >= jt.abs(dz))
        elif s == 3:
            u, v = dx / (-dy + 1e-10), -dz / (-dy + 1e-10)
            in_face = (-dy >= jt.abs(dx)) & (-dy >= jt.abs(dz))
        elif s == 4:
            u, v = dx / (dz + 1e-10), -dy / (dz + 1e-10)
            in_face = (dz >= jt.abs(dx)) & (dz >= jt.abs(dy))
        else:
            u, v = -dx / (-dz + 1e-10), -dy / (-dz + 1e-10)
            in_face = (-dz >= jt.abs(dx)) & (-dz >= jt.abs(dy))

        uv_01 = jt.stack([u * 0.5 + 0.5, v * 0.5 + 0.5], dim=-1)
        grid = (uv_01 * 2.0 - 1.0).reshape(1, N, 1, 2)
        face_nchw = tex[0, s:s+1].permute(0, 3, 1, 2)
        sampled = F.grid_sample(face_nchw, grid, mode=mode, padding_mode='border', align_corners=False)
        sampled = sampled[0, :, :, 0].permute(1, 0)
        result = result + sampled * in_face.float().unsqueeze(-1)
    return result


# ---------------------------------------------------------------------------
# Legacy compatibility stubs (unused, kept for import compatibility)
# ---------------------------------------------------------------------------

class TextureMipWrapper:
    """Stub for compatibility with old code that references this class."""
    def __init__(self, **kwargs):
        pass


def texture_construct_mip(texin, max_mip_level=None, cube_mode=False):
    """Stub: mip construction is handled by build_mips() in light.py."""
    return TextureMipWrapper()
