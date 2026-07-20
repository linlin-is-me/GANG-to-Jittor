# JGaussian-aligned: cuda_header with #include + direct CudaRasterizer::Rasterizer::* calls.
# forward_0/forward_1 split for exact buffer sizing (aligned with JGaussian Phase 2.4a).
import os
import numpy as np
import jittor as jt

# Phase 22: Global cache for CUDA backward gradients (bypass Jittor autograd crash)
GRAD_CACHE = {}

_base = os.path.dirname(os.path.abspath(__file__))
_header = os.path.join(_base, 'cuda_rasterizer')
_glm = os.path.join(_base, 'third_party', 'glm')
_lib = os.path.join(_base, 'build')

# CRITICAL: If project path contains spaces, nvcc/ld misparses ALL -I/-L flags.
# The space splits the flag argument in Jittor's compiler, even with quotes.
# Workaround: symlink/copy to /tmp/ (no spaces) and use those paths.
if ' ' in _base:
    import shutil
    _safe_root = '/tmp/jt_rasterizer'
    os.makedirs(_safe_root, exist_ok=True)
    # Copy librasterizer.so
    _safe_lib = os.path.join(_safe_root, 'build')
    os.makedirs(_safe_lib, exist_ok=True)
    _safe_so = os.path.join(_safe_lib, 'librasterizer.so')
    _src_so = os.path.join(_lib, 'librasterizer.so')
    if os.path.exists(_src_so):
        shutil.copy2(_src_so, _safe_so)
    # Symlink cuda_rasterizer headers
    _safe_header = os.path.join(_safe_root, 'cuda_rasterizer')
    if not os.path.exists(_safe_header):
        os.symlink(_header, _safe_header)
    # Symlink glm headers
    _safe_glm = os.path.join(_safe_root, 'third_party/glm')
    if not os.path.exists(_safe_glm):
        os.makedirs(os.path.dirname(_safe_glm), exist_ok=True)
        os.symlink(_glm, _safe_glm)
    _lib = _safe_lib
    _header = _safe_header
    _glm = _safe_glm

proj_options = {f'FLAGS: -I"{_header}" -I"{_glm}" -l"rasterizer" -L"{_lib}"': 1}

cuda_header = """
#include <math.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "rasterizer_impl.h"
#include <fstream>
#include <string>
#include <functional>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// ---- surface_align KNN kernels (device code) ----

__global__ void processKNNCUDA(int P, int K,
    const float* xyzs, const float* rotations, const int* indexs,
    float* mean_ds, float* out_loss_d, float* out_loss_normal)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P) return;

    int _xc = P * K;
    int _ci = indexs[K*idx];
    if (_ci < 0 || _ci >= _xc) return;

    float3 xyz_current = {xyzs[3*_ci], xyzs[3*_ci+1], xyzs[3*_ci+2]};
    float4 rotation_current = {rotations[4*_ci], rotations[4*_ci+1],
                                rotations[4*_ci+2], rotations[4*_ci+3]};
    float r = rotation_current.x, x = rotation_current.y, y = rotation_current.z, z = rotation_current.w;
    float3 normal_current = {2*(x*z+r*y), 2*(y*z-r*x), 1-2*(x*x+y*y)};

    for (int i = 0; i < K; i++) {
        int _ni = indexs[K*idx+i];
        if (_ni < 0 || _ni >= _xc) continue;

        float3 xyz = {xyzs[3*_ni], xyzs[3*_ni+1], xyzs[3*_ni+2]};
        float4 rot = {rotations[4*_ni], rotations[4*_ni+1],
                       rotations[4*_ni+2], rotations[4*_ni+3]};
        float ri=rot.x, xi=rot.y, yi=rot.z, zi=rot.w;
        float3 normal = {2*(xi*zi+ri*yi), 2*(yi*zi-ri*xi), 1-2*(xi*xi+yi*yi)};

        float cos_theta = normal_current.x*normal.x + normal_current.y*normal.y + normal_current.z*normal.z;
        if (cos_theta < 1.0f && cos_theta > 0.96593f) {
            mean_ds[_ci] += xyz.x*normal.x + xyz.y*normal.y + xyz.z*normal.z;
            out_loss_normal[_ci] += 1.0f - cos_theta;
        }
    }

    mean_ds[_ci] /= (float)K;

    for (int i = 0; i < K; i++) {
        int _ni = indexs[K*idx+i];
        if (_ni < 0 || _ni >= _xc) continue;

        float3 xyz = {xyzs[3*_ni], xyzs[3*_ni+1], xyzs[3*_ni+2]};
        float4 rot = {rotations[4*_ni], rotations[4*_ni+1],
                       rotations[4*_ni+2], rotations[4*_ni+3]};
        float ri=rot.x, xi=rot.y, yi=rot.z, zi=rot.w;
        float3 normal = {2*(xi*zi+ri*yi), 2*(yi*zi-ri*xi), 1-2*(xi*xi+yi*yi)};

        float cos_theta = normal_current.x*normal.x + normal_current.y*normal.y + normal_current.z*normal.z;
        if (cos_theta < 1.0f && cos_theta > 0.96593f) {
            float d = xyz.x*normal.x + xyz.y*normal.y + xyz.z*normal.z;
            out_loss_d[_ci] += (d - mean_ds[_ci]) * (d - mean_ds[_ci]);
        }
    }

}

__global__ void processKNNBackwardCUDA(int P, int K,
    const float* xyzs, const float* rotations, const int* indexs,
    const float* mean_ds,
    const float* grad_out_loss_d, const float* grad_out_loss_normal,
    float* dL_dxyzs, float* dL_drotations)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P) return;

    int _xc = P * K;
    int _ci = indexs[K*idx];
    if (_ci < 0 || _ci >= _xc) return;

    float4 rotation_current = {rotations[4*_ci], rotations[4*_ci+1],
                                rotations[4*_ci+2], rotations[4*_ci+3]};
    float r=rotation_current.x, x=rotation_current.y, y=rotation_current.z, z=rotation_current.w;
    float3 normal_current = {2*(x*z+r*y), 2*(y*z-r*x), 1-2*(x*x+y*y)};

    for (int i = 0; i < K; i++) {
        int _ni = indexs[K*idx+i];
        if (_ni < 0 || _ni >= _xc) continue;

        float3 xyz = {xyzs[3*_ni], xyzs[3*_ni+1], xyzs[3*_ni+2]};
        float4 rot = {rotations[4*_ni], rotations[4*_ni+1],
                       rotations[4*_ni+2], rotations[4*_ni+3]};
        float ri=rot.x, xi=rot.y, yi=rot.z, zi=rot.w;
        float3 normal = {2*(xi*zi+ri*yi), 2*(yi*zi-ri*xi), 1-2*(xi*xi+yi*yi)};

        float cos_theta = normal_current.x*normal.x + normal_current.y*normal.y + normal_current.z*normal.z;
        if (cos_theta < 1.0f && cos_theta > 0.96593f) {
            float d = xyz.x*normal.x + xyz.y*normal.y + xyz.z*normal.z;
            float dL_dout_loss_d = 2.0f * (d - mean_ds[_ci]) * grad_out_loss_d[_ci];
            atomicAdd(&(dL_dxyzs[3*_ni]),     dL_dout_loss_d * normal.x);
            atomicAdd(&(dL_dxyzs[3*_ni + 1]), dL_dout_loss_d * normal.y);
            atomicAdd(&(dL_dxyzs[3*_ni + 2]), dL_dout_loss_d * normal.z);

            float dL_dnx = dL_dout_loss_d * xyz.x - normal_current.x;
            float dL_dny = dL_dout_loss_d * xyz.y - normal_current.y;
            float dL_dnz = dL_dout_loss_d * xyz.z - normal_current.z;

            atomicAdd(&(dL_drotations[4*_ni]),     2*yi*dL_dnx - 2*xi*dL_dny);
            atomicAdd(&(dL_drotations[4*_ni + 1]), 2*zi*dL_dnx - 2*ri*dL_dny - 4*xi*dL_dnz);
            atomicAdd(&(dL_drotations[4*_ni + 2]), 2*ri*dL_dnx + 2*zi*dL_dny - 4*yi*dL_dnz);
            atomicAdd(&(dL_drotations[4*_ni + 3]), 2*xi*dL_dnx + 2*yi*dL_dny);
        }
    }
}

// ---- surface_align host wrappers ----

void launch_surface_align(int P, int K,
    const float* xyzs, const float* rotations, const int* knn_index,
    float* out_loss_d, float* out_loss_normal, float* out_mean_d)
{
    cudaMemset(out_loss_d, 0, P * K * sizeof(float));
    cudaMemset(out_loss_normal, 0, P * K * sizeof(float));
    cudaMemset(out_mean_d, 0, P * K * sizeof(float));
    processKNNCUDA<<<(P + 255) / 256, 256>>>(P, K, xyzs, rotations, knn_index,
        out_mean_d, out_loss_d, out_loss_normal);
    cudaDeviceSynchronize();
}

void launch_surface_align_backward(int P, int K,
    const float* xyzs, const float* rotations,
    const float* mean_d, const int* knn_index,
    const float* grad_out_loss_d, const float* grad_out_loss_normal,
    float* dL_dxyzs, float* dL_drotations)
{
    processKNNBackwardCUDA<<<(P + 255) / 256, 256>>>(P, K, xyzs, rotations,
        knn_index, mean_d,
        grad_out_loss_d, grad_out_loss_normal,
        dL_dxyzs, dL_drotations);
    cudaDeviceSynchronize();
}
// Phase 36: CUDA 12.6 defines cudaMemcpy as a macro (→ cudaMemcpy_ptds, static __inline__).
// --cudart=shared needs the actual dynamic symbol, so undefine the macro.
#ifdef cudaMemcpy
#undef cudaMemcpy
#endif
"""


# ---------------------------------------------------------------------------
def markVisible(means3D, viewmatrix, projmatrix):
    P = int(means3D.shape[0])
    with jt.flag_scope(compile_options=proj_options):
        out = jt.code([P], jt.bool, inputs=[means3D, viewmatrix, projmatrix],
                       cuda_header=cuda_header,
                       cuda_src='''
            int P=in0_shape0;
            if(P != 0) CudaRasterizer::Rasterizer::markVisible(P, in0_p, in1_p, in2_p, out0_p);
        ''')
        out.compile_options = proj_options
    return out


# ---------------------------------------------------------------------------
def depthToNormal(depth_map, viewmatrix, focal_x, focal_y):
    H, W = int(depth_map.shape[0]), int(depth_map.shape[1])
    with jt.flag_scope(compile_options=proj_options):
        out = jt.code([3, H, W], jt.float32, inputs=[depth_map, viewmatrix],
                       cuda_header=cuda_header,
                       cuda_src=f'''
            int H={H},W={W};
            CudaRasterizer::Rasterizer::depthToNormal(W, H, {focal_x}f, {focal_y}f, in1_p, in0_p, out0_p);
        ''')
        out.compile_options = proj_options
    return out


# ---------------------------------------------------------------------------
def compute_buffer_size(means3D, image_width, image_height):
    """Compute exact buffer sizes using CudaRasterizer::required<T>() templates.

    Phase 50: Use precise required<T>() values to match PyTorch GANG exactly.
    ImageState = N*4 (n_contrib) + N*4 (accum_alpha) + N*8 (ranges) + 128 = N*16+128
    GeometryState = required<GeometryState>(P) varies by CUDA version; use generous estimate.
    """
    P = int(means3D.shape[0])
    W = int(image_width); H = int(image_height)
    N = W * H
    # ImageState: exact formula matching required<ImageState>(N) in rasterizer_impl.h
    img = N * 16 + 128
    # GeometryState: generous, avoids cub::DeviceScan overflow
    geom = max(P * 128 + 32 * 1024 * 1024, 32 * 1024 * 1024)
    return geom, img


# ---------------------------------------------------------------------------
def RasterizeGaussiansCUDA(
    background, means3D, colors, opacity, scales, rotations,
    scale_modifier, cov3D_precomp, norm3D_precomp, extra_attrs,
    viewmatrix, projmatrix, tan_fovx, tan_fovy,
    image_height, image_width, sh, degree, campos,
    prefiltered, debug,
):
    """Simplified: single-phase rasterize using heuristic buffer sizes.
    Avoids the forward_0→host_buf→forward_1 split which is unreliable
    on non-unified-memory GPUs (RTX 3060)."""
    P, H, W = int(means3D.shape[0]), int(image_height), int(image_width)
    D = degree
    M = sh.shape[1] if sh.ndim >= 2 and sh.shape[0] > 0 else 0
    ED = extra_attrs.shape[1] if extra_attrs.ndim >= 2 and extra_attrs.shape[0] > 0 else 0
    pre = 1 if prefiltered else 0
    dbg = 1 if debug else 0
    pre_bool = prefiltered
    dbg_bool = debug

    geom_size, img_size = compute_buffer_size(means3D, W, H)
    # bin_size_val comes from forward_0 (required<BinningState>(num_rendered))

    with jt.flag_scope(compile_options=proj_options):
        # === Phase 1: forward_0 (preprocess + count) ===
        # Use jt.code float32 output tensors for scalar return values.
        # (The old host_buf volatile pointer hack is unreliable on non-unified-memory GPUs.)
        nr_out = jt.array(jt.zeros([1], dtype=jt.int32))
        bs_out = jt.array(jt.zeros([1], dtype=jt.int32))
        geomBuffer = jt.array(jt.zeros([geom_size], dtype='uint8'))
        radii = jt.array(jt.zeros([P], dtype='int32'))

        nr_out, bs_out, radii = jt.code(
            outputs=[nr_out, bs_out, radii],
            inputs=[background, means3D, colors, opacity, scales, rotations,
                    cov3D_precomp, norm3D_precomp,
                    viewmatrix, projmatrix, sh, campos, geomBuffer],
            data={'H': H, 'W': W, 'D': D, 'M': M, 'P': P,
                  'sc': scale_modifier, 'tfx': tan_fovx, 'tfy': tan_fovy,
                  'pre': pre, 'dbg': dbg},
            cuda_header=cuda_header,
            cuda_src=f'''
@alias(nr_out, out0) @alias(bs_out, out1) @alias(radii, out2)
@alias(background, in0) @alias(means3D, in1) @alias(colors, in2)
@alias(opacity, in3) @alias(scales, in4) @alias(rotations, in5)
@alias(cov3D, in6) @alias(norm3D, in7)
@alias(viewmatrix, in8) @alias(projmatrix, in9) @alias(sh, in10)
@alias(campos, in11) @alias(geomBuffer, in12)

const int P = data["P"];
int num_rendered = 0;
size_t bin_size = 1;
if (P != 0) {{
    float* _sh = (sh_shape0 != 0) ? sh_p : nullptr;
    float* _cl = (colors_shape0 != 0) ? colors_p : nullptr;
    float* _cv = (cov3D_shape0 != 0) ? cov3D_p : nullptr;
    float* _n3 = (norm3D_shape0 != 0) ? norm3D_p : nullptr;
    num_rendered = CudaRasterizer::Rasterizer::forward_0(
        geomBuffer->ptr<char>(),
        P, data["D"], data["M"],
        data["W"], data["H"],
        means3D_p, _sh, _cl, opacity_p,
        scales_p, data["sc"], rotations_p,
        _cv, _n3,
        viewmatrix_p, projmatrix_p,
        campos_p, data["tfx"], data["tfy"],
        (bool)data["pre"],
        (int*)radii_p,
        (bool)data["dbg"]);
    bin_size = CudaRasterizer::required<CudaRasterizer::BinningState>(num_rendered);
	}}
// Phase 36: cudaMemcpy macro is #undef'd in cuda_header — safe to use raw function name.
cudaMemcpy(nr_out_p, &num_rendered, sizeof(int), cudaMemcpyHostToDevice);
cudaMemcpy(bs_out_p, &bin_size, sizeof(int), cudaMemcpyHostToDevice);
''')
        for o in [nr_out, bs_out, radii]:
            o.compile_options = proj_options

        # Read scalar outputs: try var.data first, fall back to heuristic
        try:
            num_rendered_val = max(int(nr_out.data[0]), 1)
            bin_size_val = max(int(bs_out.data[0]), 1)
        except:
            num_rendered_val = P
            bin_size_val = geom_size  # use geom_size as bin_size estimate
        # Phase 62: num_rendered is total (tile,gaussian) pairs — CAN exceed P
        # at high resolutions. Removing the erroneous `num_rendered_val > P` clamp.
        if bin_size_val <= 0 or bin_size_val > (1 << 30):  # 1GB sanity cap
            bin_size_val = geom_size

        # === Phase 2: forward_1 (sort + render) ===
        binningBuffer = jt.array(jt.zeros([bin_size_val], dtype='uint8'))
        imgBuffer = jt.array(jt.zeros([img_size], dtype='uint8'))
        out_color = jt.array(jt.zeros([3, H, W]))
        out_depth = jt.array(jt.zeros([1, H, W]))
        out_opacity = jt.array(jt.zeros([1, H, W]))
        out_norm = jt.array(jt.zeros([3, H, W]))
        out_alpha = jt.array(jt.zeros([1, H, W]))
        out_extra = jt.array(jt.zeros([max(ED, 1), H, W])) if ED > 0 else jt.array(jt.zeros([1, H, W]))

        out_color, out_depth, out_opacity, out_norm, out_alpha, out_extra = jt.code(
            outputs=[out_color, out_depth, out_opacity, out_norm, out_alpha, out_extra],
            inputs=[background, means3D, colors, opacity, scales, rotations,
                    cov3D_precomp, norm3D_precomp, extra_attrs,
                    viewmatrix, projmatrix, campos,
                    geomBuffer, binningBuffer, imgBuffer, radii],
            data={'H': H, 'W': W, 'D': D, 'M': M, 'ED': ED, 'P': P,
                  'sc': scale_modifier, 'tfx': tan_fovx, 'tfy': tan_fovy,
                  'pre': pre, 'dbg': dbg, 'NR': num_rendered_val},
            cuda_header=cuda_header,
            cuda_src='''
@alias(out_color, out0) @alias(out_depth, out1) @alias(out_opacity, out2)
@alias(out_norm, out3) @alias(out_alpha, out4) @alias(out_extra, out5)
@alias(background, in0) @alias(means3D, in1) @alias(colors, in2)
@alias(opacity, in3) @alias(scales, in4) @alias(rotations, in5)
@alias(cov3D, in6) @alias(norm3D, in7) @alias(extra, in8)
@alias(viewmatrix, in9) @alias(projmatrix, in10) @alias(campos, in11)
@alias(geomBuffer, in12) @alias(binningBuffer, in13) @alias(imgBuffer, in14)
@alias(radii, in15)

const int P = data["P"];
if (P != 0) {
    float* _cl = (colors_shape0 != 0) ? colors_p : nullptr;
    float* _cv = (cov3D_shape0 != 0) ? cov3D_p : nullptr;
    float* _n3 = (norm3D_shape0 != 0) ? norm3D_p : nullptr;
    float* _ex = (extra_shape0 != 0) ? extra_p : nullptr;
    float* _oe = (out_extra_shape0 != 0 && data["ED"] > 0) ? out_extra_p : nullptr;
    CudaRasterizer::Rasterizer::forward_1(
        geomBuffer->ptr<char>(),
        binningBuffer->ptr<char>(),
        imgBuffer->ptr<char>(),
        P, data["D"], data["M"], data["ED"],
        data["NR"],
        background_p,
        data["W"], data["H"],
        _cl, _n3, _ex,
        viewmatrix_p, projmatrix_p,
        data["tfx"], data["tfy"],
        out_color_p, out_depth_p, out_opacity_p,
        out_norm_p, out_alpha_p, _oe,
        (int*)radii_p,
        (bool)data["dbg"]);
}
''')
        for o in [out_color, out_depth, out_opacity, out_norm, out_alpha, out_extra]:
            o.compile_options = proj_options

    # Detach scratch buffers so Jittor GC can free them
    geomBuffer = geomBuffer.detach()
    binningBuffer = binningBuffer.detach()
    imgBuffer = imgBuffer.detach()

    n_contrib = jt.zeros([H, W], dtype='int32')
    return (num_rendered_val, n_contrib, out_color, out_depth, out_opacity,
            out_norm, out_alpha, out_extra, radii,
            geomBuffer, binningBuffer, imgBuffer)


# ---------------------------------------------------------------------------
def RasterizeGaussiansBackwardCUDA(
    background, means3D, radii, colors, scales, rotations, extra_attrs,
    scale_modifier, cov3Ds_precomp, norm3Ds_precomp,
    viewmatrix, projmatrix, tan_fovx, tan_fovy,
    dL_dout_color, dL_dout_depth, dL_dout_norm, dL_dout_alpha, dL_dout_extra,
    sh, degree, campos,
    geomBuffer, R, binningBuffer, imageBuffer, alpha, debug,
):
    P = int(means3D.shape[0])
    H = int(dL_dout_color.shape[1])
    W = int(dL_dout_color.shape[2])
    D = degree
    M = sh.shape[1] if sh.ndim >= 2 and sh.shape[0] > 0 else 0
    ED = extra_attrs.shape[1] if extra_attrs.ndim >= 2 and extra_attrs.shape[0] > 0 else 0
    dbg = 1 if debug else 0
    NC = 3

    with jt.flag_scope(compile_options=proj_options):
        dL_dmeans2D = jt.zeros([P, 3])
        dL_dconic = jt.zeros([P, 2, 2])
        dL_dopacity = jt.zeros([P, 1])
        dL_dcolors = jt.zeros([P, NC])
        dL_ddepths = jt.zeros([P, 1])
        dL_dmeans3D = jt.zeros([P, 3])
        dL_dcov3D = jt.zeros([P, 6])
        dL_dnorm3D = jt.zeros([P, 3])
        dL_dsh = jt.zeros([P, M, 3]) if M > 0 else jt.zeros([1])
        dL_dscales = jt.zeros([P, 3])
        dL_drotations = jt.zeros([P, 4])
        dL_dextra_attrs = jt.zeros([P, max(ED, 1)]) if ED > 0 else jt.zeros([1])

        in_list = [background, means3D, radii, colors, scales, rotations, extra_attrs,
                   cov3Ds_precomp, norm3Ds_precomp,
                   viewmatrix, projmatrix,
                   dL_dout_color, dL_dout_depth, dL_dout_norm, dL_dout_alpha, dL_dout_extra,
                   sh, campos, geomBuffer, binningBuffer, imageBuffer, alpha]
        out_list = [dL_dmeans2D, dL_dconic, dL_dopacity, dL_dcolors, dL_ddepths,
                    dL_dmeans3D, dL_dcov3D, dL_dnorm3D, dL_dsh, dL_dscales,
                    dL_drotations, dL_dextra_attrs]
        for i, v in enumerate(in_list):
            assert isinstance(v, jt.Var), f"RasterizeGaussiansBackwardCUDA: in_list[{i}] is {type(v)}, expected jt.Var"
        for i, v in enumerate(out_list):
            assert isinstance(v, jt.Var), f"RasterizeGaussiansBackwardCUDA: out_list[{i}] is {type(v)}, expected jt.Var"
        outputs = jt.code(
            outputs=out_list,
            inputs=in_list,
            data={'H': H, 'W': W, 'D': D, 'M': M, 'ED': ED, 'R': R, 'P': P,
                  'sc': scale_modifier, 'tfx': tan_fovx, 'tfy': tan_fovy, 'dbg': dbg},
            cuda_header=cuda_header,
            cuda_src='''
// Raw inX_p/outX_p — no @alias to avoid name conflicts with 22+ inputs
float* sh_p = (in16_shape0 != 0) ? in16_p : nullptr;
float* cv_p = (in7_shape0 != 0) ? in7_p : nullptr;
float* n3_p = (in8_shape0 != 0) ? in8_p : nullptr;
float* ex_p = (in6_shape0 != 0) ? in6_p : nullptr;
float* gx_p = (in15_shape0 != 0) ? in15_p : nullptr;
float* dx_p = (out11_shape0 != 0 && data["ED"] > 0) ? out11_p : nullptr;
float* ds_p = (out8_shape0 != 0 && data["M"] > 0) ? out8_p : nullptr;
CudaRasterizer::Rasterizer::backward(
    data["P"], data["D"], data["M"], data["R"], data["ED"],
    in0_p, data["W"], data["H"],
    in1_p, sh_p, in3_p, in4_p, data["sc"], in5_p,
    cv_p, n3_p, ex_p, in9_p, in10_p, in17_p,
    data["tfx"], data["tfy"],
    (int*)in2_p,
    (char*)in18_p, (char*)in19_p, (char*)in20_p,
    in21_p,
    in11_p, in12_p, in13_p, in14_p, gx_p,
    out0_p, out1_p, out2_p,
    out3_p, out4_p, out5_p,
    out6_p, out7_p,
    ds_p, out9_p, out10_p,
    dx_p, (bool)data["dbg"]);
''')
        for o in outputs:
            o.compile_options = proj_options
        jt.sync()

    # Gradients exported via return value — direct_backward.py reads them via .numpy()

    return (outputs[0], outputs[3], outputs[2], outputs[5], outputs[6], outputs[7],
            outputs[8], outputs[9], outputs[10], outputs[11])


# ---------------------------------------------------------------------------
def RasterizeGaussiansFilterCUDA(
    means3D, scales, rotations, scale_modifier, cov3D_precomp,
    viewmatrix, projmatrix, tan_fovx, tan_fovy,
    image_height, image_width, prefiltered, debug,
):
    P = int(means3D.shape[0])
    H = int(image_height)
    W = int(image_width)
    M = 0
    pre = 1 if prefiltered else 0
    dbg = 1 if debug else 0
    gsz = P * 256 + 65536
    bsz = P * 128 + 65536
    isz = H * W * 32 + 4096

    with jt.flag_scope(compile_options=proj_options):
        geomBuf = jt.zeros([gsz], dtype='uint8')
        binningBuf = jt.zeros([bsz], dtype='uint8')
        imgBuf = jt.zeros([isz], dtype='uint8')
        radii = jt.zeros([P], dtype='int32')

        outputs = jt.code(
            outputs=[radii, geomBuf, binningBuf, imgBuf],
            inputs=[means3D, scales, rotations, cov3D_precomp,
                    viewmatrix, projmatrix],
            data={'H': H, 'W': W, 'P': P, 'M': M, 'sc': scale_modifier,
                  'tfx': tan_fovx, 'tfy': tan_fovy, 'pre': pre, 'dbg': dbg},
            cuda_header=cuda_header,
            cuda_src='''
@alias(radii, out0) @alias(geomBuf, out1) @alias(binningBuf, out2) @alias(imgBuf, out3)
@alias(means3D, in0) @alias(scales, in1) @alias(rotations, in2) @alias(cov3D, in3)
@alias(viewmatrix, in4) @alias(projmatrix, in5)

size_t g_off = 0, b_off = 0, i_off = 0;
auto geomFunc = [&](size_t N) -> char* {
    g_off = (g_off + 127) & ~127;
    char* p = (char*)out1_p + g_off; g_off += N; return p;
};
auto binningFunc = [&](size_t N) -> char* {
    b_off = (b_off + 127) & ~127;
    char* p = (char*)out2_p + b_off; b_off += N; return p;
};
auto imgFunc = [&](size_t N) -> char* {
    i_off = (i_off + 127) & ~127;
    char* p = (char*)out3_p + i_off; i_off += N; return p;
};

float* _cv = (cov3D_shape0 != 0) ? cov3D_p : nullptr;
CudaRasterizer::Rasterizer::visible_filter(
    geomFunc, binningFunc, imgFunc,
    data["P"], data["M"], data["W"], data["H"],
    means3D_p, scales_p, data["sc"], rotations_p, _cv,
    viewmatrix_p, projmatrix_p, data["tfx"], data["tfy"], (bool)data["pre"],
    (int*)radii_p, (bool)data["dbg"]);
''')
        for o in outputs:
            o.compile_options = proj_options
        jt.sync()

    return outputs[0]


# ---------------------------------------------------------------------------
def SurfaceAlignCUDA(xyzs, rotations, knn_index):
    P = int(knn_index.shape[0])
    K = int(knn_index.shape[1])
    # Phase 36: output arrays must hold N*K elements — kernel writes at index idx*K
    out_size = P * K

    with jt.flag_scope(compile_options=proj_options):
        out_loss_d = jt.zeros([out_size])
        out_loss_normal = jt.zeros([out_size])
        out_mean_d = jt.zeros([out_size])

        outputs = jt.code(
            outputs=[out_loss_d, out_loss_normal, out_mean_d],
            inputs=[xyzs, rotations, knn_index],
            data={'P': P, 'K': K},
            cuda_header=cuda_header,
            cuda_src='''
@alias(out_loss_d, out0) @alias(out_loss_normal, out1) @alias(out_mean_d, out2)
@alias(xyzs, in0) @alias(rotations, in1) @alias(knn_index, in2)
launch_surface_align(data["P"], data["K"],
    xyzs_p, rotations_p, (int*)knn_index_p,
    out_loss_d_p, out_loss_normal_p, out_mean_d_p);
''')
        for o in outputs:
            o.compile_options = proj_options
        jt.sync()

    return outputs[0], outputs[1], jt.zeros([0]), outputs[2]


# ---------------------------------------------------------------------------
def SurfaceAlignBackwardCUDA(
    xyzs, rotations, mean_d, knn_index,
    grad_out_loss_d, grad_out_loss_normal,
):
    P = int(knn_index.shape[0])          # N, used as thread count / kernel loop bound
    K = int(knn_index.shape[1])
    P_total = int(xyzs.shape[0])         # N*K, used for output allocation

    with jt.flag_scope(compile_options=proj_options):
        dL_dxyzs = jt.zeros([P_total, 3])
        dL_drotations = jt.zeros([P_total, 4])

        outputs = jt.code(
            outputs=[dL_dxyzs, dL_drotations],
            inputs=[xyzs, rotations, mean_d, knn_index,
                    grad_out_loss_d, grad_out_loss_normal],
            data={'P': P, 'K': K},
            cuda_header=cuda_header,
            cuda_src='''
@alias(dL_dxyzs, out0) @alias(dL_drotations, out1)
@alias(xyzs, in0) @alias(rotations, in1) @alias(mean_d, in2) @alias(knn_index, in3)
@alias(grad_loss_d, in4) @alias(grad_loss_normal, in5)
launch_surface_align_backward(data["P"], data["K"],
    xyzs_p, rotations_p, mean_d_p, (int*)knn_index_p,
    grad_loss_d_p, grad_loss_normal_p,
    dL_dxyzs_p, dL_drotations_p);
''')
        for o in outputs:
            o.compile_options = proj_options
        jt.sync()

    return outputs[0], outputs[1]


# ---------------------------------------------------------------------------
# GPU repeat utilities (ops layer — plain jt.code, no jt.Function wrapper)
# Analogous to JGaussian's mark_visible(): used within jt.no_grad() or inside
# a jt.Function's execute/grad methods.
# ---------------------------------------------------------------------------

def repeat_cuda(x, K, cols=None):
    """GPU repeat: (N, C) -> (N*K, C_out).
    If cols=(start, end), slices columns [start:end] inside the kernel (avoids
    GetitemOp on CUDA-only output). Otherwise repeats all columns."""
    K_int = int(K)
    N = x.shape[0]
    C_full = x.shape[1]
    if cols is not None:
        col_start, col_end = cols
        C_out = col_end - col_start
    else:
        col_start, col_end = 0, C_full
        C_out = C_full
    return jt.code(
        [N * K_int, C_out], jt.float32,
        inputs=[x],
        cuda_src=f"""
            __global__ void repeat_k(int N, int C_full, int col_start,
                                     const float* __restrict__ src, float* __restrict__ dst) {{
                int tid = blockIdx.x * blockDim.x + threadIdx.x;
                int total = N * {K_int} * {C_out};
                if (tid >= total) return;
                int c = tid % {C_out};
                int flat = tid / {C_out};
                int n = flat / {K_int};
                dst[tid] = src[n * C_full + col_start + c];
            }}
            int N = in0_shape0, C_full = in0_shape1;
            int blocks = (N * {K_int} * {C_out} + 255) / 256;
            repeat_k<<<blocks, 256>>>(N, C_full, {col_start}, in0_p, out0_p);
        """,
    )


def slice_cols_cuda(x, start, end):
    """GPU column slice: (N, C) -> (N, end-start). No backward — use inside jt.no_grad()."""
    C_out = end - start
    return jt.code(
        [x.shape[0], C_out], jt.float32,
        inputs=[x],
        cuda_src=f"""
            __global__ void slice_cols_k(int N, int C, int start, const float* __restrict__ src, float* __restrict__ dst) {{
                int tid = blockIdx.x * blockDim.x + threadIdx.x;
                int total = N * {C_out};
                if (tid >= total) return;
                dst[tid] = src[(tid / {C_out}) * C + start + (tid % {C_out})];
            }}
            int N = in0_shape0, C = in0_shape1;
            int blocks = (N * {C_out} + 255) / 256;
            slice_cols_k<<<blocks, 256>>>(N, C, {start}, in0_p, out0_p);
        """,
    )


def repeat_sum_bwd_cuda(grad_out, K):
    """GPU sum over K: (N*K, C) -> (N, C). Used inside _surface_align.grad()."""
    K_int = int(K)
    N = grad_out.shape[0] // K_int
    C = grad_out.shape[1]
    return jt.code(
        [N, C], jt.float32,
        inputs=[grad_out],
        cuda_src=f"""
            __global__ void repeat_sum_bwd_k(int N, int C, const float* __restrict__ src, float* __restrict__ dst) {{
                int tid = blockIdx.x * blockDim.x + threadIdx.x;
                if (tid >= N * C) return;
                int n = tid / C, c = tid % C;
                float acc = 0.0f;
                for (int k = 0; k < {K_int}; ++k) acc += src[(n * {K_int} + k) * C + c];
                dst[tid] = acc;
            }}
            int blocks = ({N} * in0_shape1 + 255) / 256;
            repeat_sum_bwd_k<<<blocks, 256>>>({N}, in0_shape1, in0_p, out0_p);
        """,
    )
