#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
from __future__ import annotations
import jittor as jt

import math
# from depth_normal_gauss import GaussianRasterizationSettings,GaussianRasterizer
# from light_geo_gauss import GaussianRasterizationSettings,GaussianRasterizer,SurfaceAlign
from light_gaussian import GaussianRasterizationSettings,GaussianRasterizer
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from scene.gaussian_model import GaussianModel  # type hint only, avoids circular import
import numpy as np
import jittor.nn as F
from utils.sh_utils import eval_sh
from utils.graphics_utils import rgb_to_srgb
# from Baking import recon_occlusion
# import open3d as o3d
from utils.graphics_utils import normal_from_depth_image
from utils.loss_utils import eikonal_loss
from utils.jt_safe import path_log

def debug_hook(module, input, output):
    if jt.isnan(output).any():
        print(f"NaN detected in {module.__class__.__name__}")
        print("Input range:", input[0].min(), input[0].max())
        print("Output range:", output.min(), output.max())
        raise ValueError("NaN encountered")
    

def build_rotation(r):
    norm = jt.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )

    q = r / norm[:, None]

    R = jt.zeros((q.size(0), 3, 3))

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R

def local_var(inputs):
    # input N M C
    aa = jt.var(inputs,dim=1)

    return jt.mean(jt.sum(aa,dim=-1))

def local_var_normal(inputs,mask):
    # input N M C
    unique_classes = mask.unique()
    variances = []
    for cls in unique_classes:

        cls_mask = (mask == cls).squeeze()
        cls_data = inputs[cls_mask]

        if cls_data.size(0) > 1:  
            cls_variance = cls_data.var(dim=1, unbiased=False)
            variances.append(cls_variance)
    if variances:
        variances = jt.concat(variances, dim=0)  
        overall_mean_variance = variances.mean()  

    return overall_mean_variance



def _bool_to_indices(mask):
    """Normalize mask/indices to int indices via GPU jt.nonzero() — no CPU round-trip.

    Jittor 1.3.11: jt.nonzero() uses where_op.cc CUDA kernel (warp/block/CUB).
    Only numpy bool arrays are handled on CPU (no GPU round-trip needed).
    """
    if mask is None:
        return None
    # Already int indices (numpy)
    if isinstance(mask, np.ndarray) and mask.dtype in (np.int32, np.int64):
        return mask
    # Numpy bool mask — trivially convert on CPU (already on CPU)
    if isinstance(mask, np.ndarray):
        return np.nonzero(mask)[0]
    # Jittor boolean mask → GPU nonzero (where_op.cc CUDA kernel)
    try:
        idx = jt.nonzero(mask)  # → [M] or [M, 1] jt.Var on GPU
        if idx.ndim > 1:
            idx = idx.squeeze(1)
        return idx  # jt.Var — preserves autograd for downstream
    except:
        # Last resort: all indices (no GPU→CPU round-trip)
        return np.arange(mask.shape[0], dtype=np.int32)


_safe_index_logged = {"numpy": False, "jtcode": False, "arange": False, "jtidx": False}
_safe_index_stats = {"jtidx": 0, "jtterr": 0, "jtcode": 0, "numpy": 0}

# Cache for _gather_jt compiled kernels (by C value)
_gather_jt_cache = {}

def _gather_jt(tensor, idx):
    """Gradient-safe gather: tensor[idx] via jt.code CUDA kernel.

    Unlike jt.array(tensor.numpy()[idx]), jt.code with inputs= preserves
    the autograd connection. Jittor traces the kernel and generates
    scatter-add backward automatically.

    Args:
        tensor: jt.Var [M, C]
        idx: numpy int array [N] — indices to gather
    Returns:
        jt.Var [N, C] — gathered tensor with autograd preserved
    """
    N = len(idx)
    if N == 0:
        return tensor[:0]
    C = tensor.shape[1]
    idx_jt = jt.array(idx.astype(np.int32))

    # Use cached kernel template for each C value
    cache_key = (C, tensor.dtype)
    if cache_key not in _gather_jt_cache:
        kernel = f'''
        __global__ void gather_kernel_{C}(float* out, float* inp, int* idx, int N) {{
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            if (tid >= N) return;
            int src = idx[tid];
            for (int c = 0; c < {C}; c++) {{
                out[tid * {C} + c] = inp[src * {C} + c];
            }}
        }}
        '''
        _gather_jt_cache[cache_key] = kernel
    else:
        kernel = _gather_jt_cache[cache_key]

    # jt.code with inputs= registers tensor in autograd graph
    out = jt.code([N, C], tensor.dtype, [tensor, idx_jt],
        cuda_src=str(N) + ' ' + kernel,
    )
    return out

def _safe_index(tensor, idx):
    """Integer indexing: uses Jittor native (preserves autograd), jt.code gather as fallback.

    Jittor 1.3.11: integer indexing with jt.Var or numpy int arrays preserves the autograd
    graph via GPU gather kernel (GetitemOp). No numpy round-trip — never breaks gradients.

    _bool_to_indices already converts booleans to int indices, so boolean mask crash is avoided.
    """
    if isinstance(idx, jt.Var):
        n = idx.shape[0]
    else:
        n = len(idx)
    if n == 0:
        return tensor[:0]
    if isinstance(idx, np.ndarray) and n == tensor.shape[0] and idx[0] == 0 and idx[-1] == n - 1:
        if not _safe_index_logged["arange"]:
            path_log("[data_src] _safe_index: arange shortcut (no copy)")
            _safe_index_logged["arange"] = True
        return tensor

    # Tier 1: Jittor native integer indexing — preserves autograd graph (GetitemOp CUDA kernel)
    try:
        result = tensor[idx]
        _safe_index_stats["jtidx"] += 1
        if not _safe_index_logged["jtidx"]:
            path_log("[data_src] _safe_index: Jittor native int indexing (preserves grad)")
            _safe_index_logged["jtidx"] = True
        return result
    except Exception as _e_idx:
        _safe_index_stats["jtterr"] += 1
        _msg = str(_e_idx)[:100]
        if _safe_index_stats["jtterr"] <= 3:
            path_log(f"[WARN] _safe_index: Jittor int indexing FAILED for {tensor.shape}, "
                     f"falling back to jt.code gather: {_msg}")

    # Tier 2: jt.code gather (preserves autograd via jt.code inputs= mechanism)
    try:
        result = _gather_jt(tensor, idx)
        _safe_index_stats["jtcode"] += 1
        if not _safe_index_logged["jtcode"]:
            path_log("[data_src] _safe_index: jt.code gather (preserves grad)")
            _safe_index_logged["jtcode"] = True
        return result
    except Exception as _e_jt:
        _msg2 = str(_e_jt)[:100]
        _safe_index_stats["numpy"] += 1
        path_log(f"[ERROR] _safe_index: ALL methods FAILED: {_msg2}")
        raise RuntimeError(f"_safe_index: all fallback methods failed for tensor "
                          f"shape={tensor.shape}: {_msg2}")



def generate_neural_gaussians(viewpoint_camera, pc : GaussianModel, visible_mask=None, is_training=False, iteration= 0, ape_code=-1, is_pbr=False):
    ## view frustum filtering for acceleration
    global roughness, albedo, matallic
    indices = _bool_to_indices(visible_mask)

    def _idx(tensor):
        return tensor if indices is None else _safe_index(tensor, indices)

    anchor = _idx(pc.get_anchor)
    feat = _idx(pc.get_anchor_feat)
    level = _idx(pc.get_level)
    # For offset-level tensors, indices are anchor-level — need expansion
    if indices is not None:
        K = pc.n_offsets
        # Phase 78: support both numpy int and jt.Var int indices
        if isinstance(indices, np.ndarray):
            offset_indices = np.repeat(indices * K, K) + np.tile(np.arange(K), len(indices))
        else:
            # jt.Var int indices from jt.nonzero — use GPU repeat (safe in inference)
            n = indices.shape[0]
            offset_indices = (indices * K).unsqueeze(1).repeat(1, K) + jt.arange(K)
            offset_indices = offset_indices.reshape(-1)
    else:
        offset_indices = None
    # _offset is [N*K, 3] — already flat, no reshape needed
    grid_offsets = pc._offset
    if indices is not None:
        grid_offsets = _safe_index(grid_offsets, offset_indices)
    # _scaling is anchor-level [N, 6]
    grid_scaling = _idx(pc.get_scaling)

    sdf_loss = 0

    local_loss = 0
    ## get view properties for anchor
    ob_view = anchor - viewpoint_camera.camera_center
    # dist
    ob_dist_raw = ob_view.norm(dim=1, keepdim=True)
    # Normalize distance to ~1.0 to prevent MLP saturation on large-scale scenes
    ob_dist = ob_dist_raw / (ob_dist_raw.mean().detach() + 1e-8)
    # view direction (unit vector)
    ob_view = ob_view / ob_dist_raw

    ## view-adaptive feature
    if pc.use_feat_bank:
        if pc.add_level:
            cat_view = jt.concat([ob_view, level], dim=1)
        else:
            cat_view = ob_view
        
        bank_weight = pc.get_featurebank_mlp(cat_view).unsqueeze(dim=1) # [n, 1, 3]

        ## multi-resolution feat
        feat = feat.unsqueeze(dim=-1)
        feat = feat[:,::4, :1].repeat([1,4,1])*bank_weight[:,:,:1] + \
            feat[:,::2, :1].repeat([1,2,1])*bank_weight[:,:,1:2] + \
            feat[:,::1, :1]*bank_weight[:,:,2:]
        feat = feat.squeeze(dim=-1) # [n, c]

    if pc.add_level:
        cat_local_view = jt.concat([feat, ob_view, ob_dist, level], dim=1) # [N, c+3+1+1]
        cat_local_view_wodist = jt.concat([feat, ob_view, level], dim=1) # [N, c+3+1]
    else:
        cat_local_view = jt.concat([feat, ob_view, ob_dist], dim=1) # [N, c+3+1]
        cat_local_view_wodist = jt.concat([feat, ob_view], dim=1) # [N, c+3]

    if pc.appearance_dim > 0:
        if ape_code < 0:
            camera_indicies = jt.ones(cat_local_view[:,0].shape, dtype=jt.int64) * viewpoint_camera.uid
            appearance = pc.get_appearance(camera_indicies)
        else:
            camera_indicies = jt.ones(cat_local_view[:,0].shape, dtype=jt.int64) * ape_code[0]
            appearance = pc.get_appearance(camera_indicies)


    # get offset's opacity
    if pc.add_opacity_dist:
        _opacity_input = cat_local_view
        neural_opacity = pc.get_opacity_mlp(cat_local_view) # [N, k]
    else:
        _opacity_input = cat_local_view_wodist
        neural_opacity = pc.get_opacity_mlp(cat_local_view_wodist)
    
    if pc.dist2level=="progressive":
        prog = _idx(pc._prog_ratio)
        transition_mask = _idx(pc.transition_mask)
        # prog[~transition_mask] = 1.0  # disabled: boolean setitem uses jt.where (CUDA-only)
        neural_opacity = neural_opacity * prog

    # opacity mask generation
    neural_opacity = neural_opacity.reshape([-1, 1])
    mask = (neural_opacity>0.0)  # Phase 42: matches PyTorch Tanh threshold (Tanh outputs [-1,1], ~50% > 0)
    mask = mask.view(-1)


    # select opacity
    mask_indices = _bool_to_indices(mask)
    opacity = _safe_index(neural_opacity, mask_indices)

    # get offset's color
    if pc.appearance_dim > 0:
        if pc.add_color_dist:
            _color_input = jt.concat([cat_local_view, appearance], dim=1)
            color = pc.get_color_mlp(_color_input)
        else:
            _color_input = jt.concat([cat_local_view_wodist, appearance], dim=1)
            color = pc.get_color_mlp(_color_input)
    else:
        if pc.add_color_dist:
            _color_input = cat_local_view
            color = pc.get_color_mlp(cat_local_view)
        else:
            _color_input = cat_local_view_wodist
            color = pc.get_color_mlp(cat_local_view_wodist)


    # offset's color: already in [N, K*3=30] → reshape to [N*K, 3]
    color = color.reshape([anchor.shape[0]*pc.n_offsets, 3])

    # get offset's cov
    if pc.add_cov_dist:
        _cov_input = cat_local_view
        scale_rot = pc.get_cov_mlp(cat_local_view)
    else:
        _cov_input = cat_local_view_wodist
        scale_rot = pc.get_cov_mlp(cat_local_view_wodist)
    scale_rot = scale_rot.reshape([anchor.shape[0]*pc.n_offsets, 7])

    # offsets
    offsets = grid_offsets.view([-1, 3]) # [mask]

    grid_rotation = _idx(pc._rotation)

    # Build individual repeated tensors via index-based expansion.
    # Replace unsqueeze+repeat+reshape — Jittor 1.3.11's repeat backward
    # lowers to binary_op between [N,K,C] and [N,C], causing shape mismatch
    # (xshape(10) != yshape(1448)). Integer-index backward uses scatter-add, no 3D intermediate.
    K = pc.n_offsets
    N_anchor = anchor.shape[0]
    # Use numpy int array as index (NOT jt.Var): avoids autograd tracking of index.
    # Jittor's tensor[numpy_int_array] forward is a gather, backward is scatter-add
    # without intermediate 3D tensors (unlike repeat/gather with jt.Var index).
    _expand_np = np.arange(N_anchor, dtype=np.int32).repeat(K)
    scaling_expanded = grid_scaling[_expand_np]             # [N*K, 6]
    rotation_expanded = grid_rotation[_expand_np]           # [N*K, 4]
    repeat_anchor = anchor[_expand_np]                      # [N*K, 3]

    if pc.normal_detal and iteration>500:
        flag = 1
    elif iteration>3000:
        flag = 1
    else:
        flag=0

    if is_training and flag ==1:

        # Phase 54d: Pure Jittor SurfaceAlign — no jt.code/jt.Function, autograd-native.
        # Replaces SurfaceAlignCUDA kernel. jt.grad() CAN trace through pure Jittor ops.
        # _expand_np = np.arange(N, dtype=int32).repeat(K) — already computed at line 401.
        K = pc.n_offsets
        scaling_3_repeat = grid_scaling[_expand_np, :3]   # [N*K, 3] numpy-indexed gather
        rot_input = scale_rot[:, 3:7]                      # [N*K, 4] pure Jittor slice
        offsets_all = offsets * scaling_3_repeat
        rot_all = pc.rotation_activation(rot_input)

        # Compute xyz from anchor expansion
        repeat_anchor_all = anchor[_expand_np]              # [N*K, 3]
        xyz_all = repeat_anchor_all + offsets_all            # [N*K, 3]

        # ---- Pure Jittor SurfaceAlign (replaces SurfaceAlignCUDA) ----
        # Math identical to processKNNCUDA in rasterize_points_jt.py:62-115
        r, x, y, z = rot_all[:,0], rot_all[:,1], rot_all[:,2], rot_all[:,3]
        normals = jt.stack([
            2*(x*z + r*y),
            2*(y*z - r*x),
            1 - 2*(x*x + y*y)
        ], dim=-1)  # [N*K, 3]

        xyzs_NK = xyz_all.reshape(-1, K, 3)
        norms_NK = normals.reshape(-1, K, 3)

        # Center = first Gaussian per anchor  [N, 3]
        center_xyz = xyzs_NK[:, 0, :]
        center_norm = norms_NK[:, 0, :]

        # cos between center normal and all K neighbor normals  [N, K]
        cos_theta = (center_norm.unsqueeze(1) * norms_NK).sum(dim=-1)

        # Mask: cos in (0.96593, 1.0) — within ~15 degrees
        mask = (cos_theta < 1.0) & (cos_theta > 0.96593)
        mask_f = mask.float32()

        # Signed distance: dot(xyz_i, center_normal)  [N, K]
        dists = (xyzs_NK * center_norm.unsqueeze(1)).sum(dim=-1)

        # Mean signed distance per center, masked
        count = mask_f.sum(dim=-1).clamp(min_v=1.0)  # [N]
        mean_d = (dists * mask_f).sum(dim=-1) / count  # [N]

        # Per-center normal loss  [N]
        pair_normal_center = ((1.0 - cos_theta) * mask_f).sum(dim=-1)

        # Per-center distance variance loss  [N]
        pair_d_center = (((dists - mean_d.unsqueeze(1)) ** 2) * mask_f).sum(dim=-1)

        # Scatter to [N*K] (loss at center position 0, K, 2K, ...)
        # Use jt.scatter: 2D output, 1D index, 2D src
        N_anchor = xyzs_NK.shape[0]
        center_idx = jt.arange(N_anchor) * K  # [N] GPU
        out_NK_2d = jt.zeros((N_anchor * K, 1))
        pair_d_nk = jt.scatter(out_NK_2d, 0, center_idx,
                               pair_d_center.reshape(-1, 1), reduce='add')
        pair_n_nk = jt.scatter(out_NK_2d, 0, center_idx,
                               pair_normal_center.reshape(-1, 1), reduce='add')

        pair_d_loss = pair_d_nk.reshape(-1)
        pair_normal_loss = pair_n_nk.reshape(-1)
        local_loss += 0.05*jt.mean(pair_d_loss) + 0.01*jt.mean(pair_normal_loss)


    # Filter each component individually via _safe_index (no concat, no split, no numpy).
    # _safe_index with arange indices is a no-op (returns tensor directly).
    # All tensors stay as jt.Var — no CPU conversion needed.
    # The rasterizer (jt.code CUDA kernel) receives these directly.
    scaling_repeat = _safe_index(scaling_expanded, mask_indices)
    rotation_repeat = _safe_index(rotation_expanded, mask_indices)
    repeat_anchor = _safe_index(repeat_anchor, mask_indices)
    color_filtered = _safe_index(color, mask_indices)      # [N*K, 3]
    scale_rot_filtered = _safe_index(scale_rot, mask_indices)  # [N*K, 7]
    offsets_filtered = _safe_index(offsets, mask_indices)     # [N*K, 3]
    

    # post-process cov (using filtered versions passed to rasterizer)
    scaling = scaling_repeat[:,3:] * jt.sigmoid(scale_rot_filtered[:,:3])
    scaling = scaling.maximum(1e-8).minimum(1.0)  # runtime clamp: prevent giant Gaussians (Phase 44)

    rot = pc.rotation_activation(rotation_repeat*scale_rot_filtered[:,3:7])

    # post-process offsets to get centers for gaussians
    offsets_out = offsets_filtered * scaling_repeat[:,:3]
    xyz = repeat_anchor + offsets_out

    # === Manual grad diagnostic capture (Phase 22) ===
    # Store intermediate tensors for manual gradient computation.
    # These are read via cupy in utils/manual_grad.py:compute_geometry_grads()
    pc._diag_data = {
        'scaling_repeat': scaling_repeat,        # [G, 6]
        'rotation_repeat': rotation_repeat,      # [G, 4]
        'scale_rot_filtered': scale_rot_filtered,  # [G, 7]
        'offsets_filtered': offsets_filtered,    # [G, 3]
        'grid_scaling': grid_scaling,            # [M, 6]
        'grid_rotation': grid_rotation,          # [M, 4]
        'anchor_M': anchor,                      # [M, 3] (before expand)
        'mask_indices': mask_indices,            # [G] numpy int64
        'visible_indices': indices,              # [M] numpy int64 (may be None)
        'expand_np': _expand_np,                 # [M*K] numpy int32
        'offsets_MK': offsets,                   # [M*K, 3]
        'scale_rot_MK': scale_rot,               # [M*K, 7] (MLP output, pre-filter)
        'offset_indices': offset_indices if 'offset_indices' in dir() else None,  # [M*K] numpy
        # Phase 30: MLP backward inputs
        'mlp_opacity_input': _opacity_input if '_opacity_input' in dir() else None,
        'mlp_cov_input': _cov_input if '_cov_input' in dir() else None,
        'mlp_color_input': _color_input if '_color_input' in dir() else None,
        # Phase 39: PBR MLP backward inputs
        'is_pbr': is_pbr,
        'mlp_pbr_input': _opacity_input if '_opacity_input' in dir() else None,
    }

    view_dir = xyz - viewpoint_camera.camera_center.repeat(xyz.shape[0], 1)
    view_dir_normal = (view_dir/view_dir.norm(dim=1, keepdim=True)).detach() # (N, 3)

    if pc.normal_detal:
        if pc.add_opacity_dist:
            delta_normal1 = pc.get_normal1_mlp(cat_local_view)  # [N, k]
            delta_normal2 = pc.get_normal2_mlp(cat_local_view)
        else:
            delta_normal1 = pc.get_normal1_mlp(cat_local_view_wodist)
            delta_normal2 = pc.get_normal2_mlp(cat_local_view_wodist)
        delta_normal1 =delta_normal1.reshape([anchor.shape[0]*pc.n_offsets, 3])
        delta_normal2 =delta_normal2.reshape([anchor.shape[0]*pc.n_offsets, 3])
        normal,delta_normal = pc.computeNorm(scaling, rot,view_dir_normal, delta_normal1,delta_normal2)
        delta_normal_norm = delta_normal.norm(dim=1, keepdim=True)*0.1
    else:
        normal = pc.computeNorm(scaling, rot,view_dir_normal)
        delta_normal_norm = None


    # Phase 93: sync to materialize non-PBR MLP outputs (opacity+color+cov+normal)
    # before PBR MLPs (roughness+albedo+metallic) start building their lazy graph.
    # This mirrors PyTorch eager execution — each MLP group executes and frees
    # intermediates independently, reducing peak lazy-graph memory by ~40%.
    jt.sync_all(); jt.gc(); jt.gc()

    if is_pbr:
        matallic = None
        if pc.add_opacity_dist:
            roughness = pc.get_roughness_mlp(cat_local_view)  # [N, k]
            albedo = pc.get_albedo_mlp(cat_local_view)
            if pc.with_matallic:
                matallic = pc.get_matallic_mlp(cat_local_view)
        else:
            roughness = pc.get_roughness_mlp(cat_local_view_wodist)
            albedo = pc.get_albedo_mlp(cat_local_view_wodist)
            if pc.with_matallic:
                matallic = pc.get_matallic_mlp(cat_local_view_wodist)

        if is_training:
        
            albedo_loss = local_var(albedo.reshape([anchor.shape[0],pc.n_offsets, 3]))
            roughness_loss = local_var(roughness.reshape([anchor.shape[0],pc.n_offsets, 1]))
            if pc.with_matallic:
                metrics_loss = local_var(matallic.reshape([anchor.shape[0],pc.n_offsets, 1]))
            if pc.with_matallic:
                local_loss += albedo_loss+roughness_loss+metrics_loss
            else:
                local_loss += albedo_loss+roughness_loss
    
        albedo = albedo.reshape([anchor.shape[0]*pc.n_offsets, 3])
        roughness = roughness.reshape([-1, 1])
        if pc.with_matallic:
            matallic = matallic.reshape([-1,1])

        albedo = _safe_index(albedo, mask_indices)
        roughness = _safe_index(roughness, mask_indices)
        if pc.with_matallic:
            matallic = _safe_index(matallic, mask_indices)

        albedo =  jt.clamp(albedo, 0.0, 1.0)
        roughness =  jt.clamp(roughness, 0.001, 1.0)
        matallic =  jt.clamp(matallic, 0.0, 1.0)

    else:
        albedo = None
        roughness = None
        matallic = None



    # Use filtered versions for rasterizer output
    color = color_filtered
    scale_rot = scale_rot_filtered

    return xyz, color, opacity, scaling, rot, neural_opacity, mask, albedo, roughness, matallic,normal,delta_normal_norm,local_loss,sdf_loss
   


def scale_loss(scaling):

    _, sorted_scale = jt.argsort(scaling, dim=-1)
    min_scale_loss = sorted_scale[...,0]
    loss_scale = 100.0*min_scale_loss.mean()

    return loss_scale


def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : jt.Var, scaling_modifier=1.0, visible_mask=None,is_pbr=False,light=None, retain_grad=False, is_training =True, Local_pkg=None,iteration = 0,ape_code=-1):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    # if is_training:
    xyz, color, opacity, scaling, rot, neural_opacity, mask, albedo, roughness, matallic,normal,delta_normal_norm,local_loss,sdf_loss = generate_neural_gaussians(viewpoint_camera, pc,visible_mask,is_training=is_training,is_pbr=is_pbr,iteration= iteration,ape_code = ape_code)

    loss_scale = scale_loss(scaling)
    if pc.normal_detal:
        delta_normal_norm = delta_normal_norm.repeat(1, 3)


    screenspace_points = jt.zeros_like(xyz) + 0
    screenspace_points.requires_grad = True
    if retain_grad:
        try:
            screenspace_points.retain_grad()
        except:
            pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)



    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    if is_pbr:

        viewdirs = jt.normalize(viewpoint_camera.camera_center - xyz, p=2, dim=-1)

        # Phase 35: build_mips only once (JIT compiles 41+ fused CUDA kernels, >60s)
        if not getattr(light, '_mips_built', False):
            light.build_mips()
            light._mips_built = True
        
        normal_t = normal * 0.5 + 0.5

        light_color, extras = light.lightRender(xyz, normal_t, albedo, roughness, matallic, viewdirs)

        # Phase 39: save lightRender inputs for PBR MLP gradient computation
        # Phase 78: gated behind is_training — not needed for inference
        if is_training:
            pc._diag_data['xyz_light'] = xyz.numpy()
            pc._diag_data['normal_t_light'] = normal_t.numpy()
            pc._diag_data['albedo_light'] = albedo.numpy()
            pc._diag_data['roughness_light'] = roughness.numpy()
            _matallic_np = matallic.numpy() if (matallic is not None) else None
            pc._diag_data['metallic_light'] = _matallic_np
            pc._diag_data['viewdirs_light'] = viewdirs.numpy()
            pc._diag_data['with_matallic'] = pc.with_matallic
            pc._diag_data['light_obj'] = light

        if is_training:
            normal = normal @ viewpoint_camera.world_view_transform[:3, :3]
        normal = normal * 0.5 + 0.5

        if pc.with_matallic:
            if pc.normal_detal:
                features = jt.concat([normal,delta_normal_norm,albedo,roughness,matallic],dim=-1)             
            else:
                features = jt.concat([normal,albedo,roughness,matallic],dim=-1)
        else:
            if pc.normal_detal:
                features = jt.concat([normal,delta_normal_norm,albedo,roughness],dim=-1)             
            else:
                features = jt.concat([normal,albedo,roughness],dim=-1)              

        color = light_color
    else:
        if is_training:
            normal = normal @ viewpoint_camera.world_view_transform[:3, :3]
        normal = normal * 0.5 + 0.5

        if pc.normal_detal:
            features = jt.concat([normal,delta_normal_norm],dim=-1)
        else:
            features = normal


    n_contri,rendered_image, rendered_depth,rendered_opacity, rendered_norm,depth_normal, rendered_alpha, radii, rendered_features = rasterizer(
        means3D=xyz,
        means2D=screenspace_points,
        shs=None,
        colors_precomp=color,
        opacities=opacity,
        scales=scaling,
        rotations=rot,
        cov3Ds_precomp=None,
        extra_attrs=features
    )
    # Store reference for manual backward (Phase 22)
    import gaussian_renderer as _gr
    _gr._last_rasterize_func = getattr(rasterizer, '_last_rasterize_func', None)
    _gr._last_rasterizer = rasterizer
    feature_dict = {}

    if is_pbr:
        if pc.with_matallic:
            if pc.normal_detal:
                precomput_normal,delta_normal_t,rendered_albedo,rendered_roughness,rendered_matallic = rendered_features.split([3,3,3,1,1], dim=0)
            else:
                precomput_normal,rendered_albedo,rendered_roughness,rendered_matallic = rendered_features.split([3,3,1,1], dim=0)
                delta_normal_t = None

            feature_dict.update({"albedo": rendered_albedo,
                            "roughness": rendered_roughness,
                            "matallic": rendered_matallic
                            })
        else:
            if pc.normal_detal:
                precomput_normal,delta_normal_t,rendered_albedo,rendered_roughness = rendered_features.split([3,3,3,1], dim=0)             
            else:
                precomput_normal,rendered_albedo,rendered_roughness = rendered_features.split([3,3,1], dim=0)
                delta_normal_t = None
            feature_dict.update({"albedo": rendered_albedo,
                            "roughness": rendered_roughness
                            })             
    else:
        if pc.normal_detal:
            precomput_normal,delta_normal_t = rendered_features.split([3, 3],dim=0)
        else:
            precomput_normal = rendered_features
            delta_normal_t = None
            

    precomput_normal = (precomput_normal - 0.5) * 2.0
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # if is_training:
    if is_pbr:
        results = {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "neural_opacity": neural_opacity,
            "selection_mask": _bool_to_indices(mask),  # numpy int idx (avoids jt.where)
            "scaling": scaling,
            "normal": rendered_norm,
            "precomput_normal": precomput_normal,
            "delta_normal":delta_normal_t,
            "depth_normal":depth_normal,
            "depth": rendered_depth,
            "opacity": rendered_opacity,
            "alpha": rendered_alpha,
            "local_loss": local_loss,
            "scale_loss":loss_scale,
            "sdf_loss":sdf_loss,
            "points":xyz,
            "points_normal":normal,
            "_diag_data": pc._diag_data if hasattr(pc, '_diag_data') else {},
            }
        results.update(feature_dict)

        return results        
        
    else:
        return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "selection_mask": _bool_to_indices(mask),  # numpy int idx (avoids jt.where)
            "neural_opacity": neural_opacity,
            "scaling": scaling,
            "normal": rendered_norm,
            "precomput_normal": precomput_normal,
            "delta_normal":delta_normal_t,
            "depth_normal":depth_normal,
            "depth": rendered_depth,
            "opacity": rendered_opacity,
            "alpha": rendered_alpha,
            "local_loss": local_loss,
            "scale_loss":loss_scale,
            "sdf_loss":sdf_loss,
            "points":xyz,
            "points_normal":normal,
            "_diag_data": pc._diag_data if hasattr(pc, '_diag_data') else {},

            }


def prefilter_voxel(viewpoint_camera, pc : GaussianModel, pipe, bg_color : jt.Var,anchor_mask=None, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)


    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    if anchor_mask is None:
        # Phase 78: GPU boolean indexing (aligns with PyTorch GANG).
        # jt.where(CUDA) + contrib.getitem handle boolean indexing on GPU correctly.
        anchor_mask = pc._anchor_mask  # GPU bool tensor

    # GPU boolean indexing — no numpy, no CPU roundtrip
    means3D = pc.get_anchor[anchor_mask]
    scales = pc.get_scaling[anchor_mask]
    rotations = pc.get_rotation[anchor_mask]

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)

    radii_pure = rasterizer.visible_filter(means3D = means3D,
        scales = scales[:,:3],
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # GPU boolean setitem (Phase 78: contrib.setitem handles bool masks on GPU)
    visible_mask = anchor_mask.clone()
    from jittor.contrib import setitem as _contrib_setitem
    _contrib_setitem(visible_mask, anchor_mask, radii_pure > 0)
    return visible_mask  # GPU bool tensor
