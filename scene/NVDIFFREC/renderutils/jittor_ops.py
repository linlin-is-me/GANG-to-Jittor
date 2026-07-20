# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Jittor migration: replaces torch.autograd.Function + CUDA plugin with pure Python/Jittor.
#
# Original ops.py used C++/CUDA plugins via torch.utils.cpp_extension.
# This file provides equivalent functionality using pure Python/Jittor.
# For BSDF functions, the existing bsdf.py already has complete Jittor implementations.
# For cubemap filtering: diffuse_cubemap uses numpy; specular_cubemap uses CUDA jt.code (Phase 38).

import numpy as np
import jittor as jt
import jittor.nn as F
import math

from .bsdf import *
from .loss import *


# ----------------------------------------------------------------------------
# Vector utility helpers (used by cubemap functions)
# ----------------------------------------------------------------------------

def _cube_to_dir(s, x, y, N):
    """Convert cube face pixel coordinates to direction vector."""
    fx = 2.0 * ((x + 0.5) / N) - 1.0
    fy = 2.0 * ((y + 0.5) / N) - 1.0
    if s == 0:
        rx, ry, rz = 1.0, -fy, -fx
    elif s == 1:
        rx, ry, rz = -1.0, -fy, fx
    elif s == 2:
        rx, ry, rz = fx, 1.0, fy
    elif s == 3:
        rx, ry, rz = fx, -1.0, -fy
    elif s == 4:
        rx, ry, rz = fx, -fy, 1.0
    elif s == 5:
        rx, ry, rz = -fx, -fy, -1.0
    else:
        raise ValueError(f"Invalid cube face: {s}")
    norm = math.sqrt(rx * rx + ry * ry + rz * rz)
    return rx / norm, ry / norm, rz / norm


def _pixel_area(x, y, N):
    """Compute solid angle of a cubemap pixel."""
    if N <= 1:
        return 1.0
    H = N // 2
    x = abs(x - H)
    y = abs(y - H)
    dx = math.atan((x + 1.0) / H) - math.atan(x / H)
    dy = math.atan((y + 1.0) / H) - math.atan(y / H)
    return dx * dy


# ----------------------------------------------------------------------------
# Precompute lookup tables for cubemap filtering (shared across calls)
# ----------------------------------------------------------------------------

_cubemap_dir_cache = {}
_cubemap_area_cache = {}


def _get_cubemap_dirs(N):
    """Get or compute direction vectors for all cubemap pixels."""
    if N not in _cubemap_dir_cache:
        dirs = np.zeros((6, N, N, 3), dtype=np.float32)
        for s in range(6):
            for y in range(N):
                for x in range(N):
                    rx, ry, rz = _cube_to_dir(s, x, y, N)
                    dirs[s, y, x, 0] = rx
                    dirs[s, y, x, 1] = ry
                    dirs[s, y, x, 2] = rz
        _cubemap_dir_cache[N] = dirs
    return _cubemap_dir_cache[N]


def _get_cubemap_areas(N):
    """Get or compute pixel areas for cubemap resolution N."""
    if N not in _cubemap_area_cache:
        areas = np.zeros((N, N), dtype=np.float32)
        for y in range(N):
            for x in range(N):
                areas[y, x] = _pixel_area(x, y, N)
        _cubemap_area_cache[N] = areas
    return _cubemap_area_cache[N]


# ----------------------------------------------------------------------------
# Diffuse cubemap: cosine-weighted integration over the entire cubemap
# ----------------------------------------------------------------------------

def diffuse_cubemap(cubemap):
    """Compute diffuse irradiance cubemap via cosine-weighted integration."""
    assert cubemap.shape[0] == 6 and cubemap.shape[1] == cubemap.shape[2], \
        f"Bad shape for cubemap tensor: {cubemap.shape}"

    res = cubemap.shape[1]
    cm_np = cubemap.numpy()

    dirs = _get_cubemap_dirs(res)
    areas = _get_cubemap_areas(res)

    src_flat = cm_np.reshape(6 * res * res, 3)
    dir_flat = dirs.reshape(6 * res * res, 3)
    area_flat = areas.reshape(1, res * res)

    output = np.zeros((6, res, res, 3), dtype=np.float32)

    for s_out in range(6):
        for y_out in range(res):
            for x_out in range(res):
                rx, ry, rz = _cube_to_dir(s_out, x_out, y_out, res)
                N = np.array([rx, ry, rz], dtype=np.float32)

                dots = np.dot(dir_flat, N)
                dots = np.clip(dots, 0.0, 0.999)
                weights = dots.reshape(6, res, res) * areas[None, :, :] / np.pi

                for s_in in range(6):
                    for c in range(3):
                        output[s_out, y_out, x_out, c] += np.sum(
                            cm_np[s_in, :, :, c] * weights[s_in])

    return jt.array(output, dtype=jt.float32)


# ----------------------------------------------------------------------------
# Specular cubemap: GGX-filtered integration — CUDA jt.code (Phase 38)
# ----------------------------------------------------------------------------

def _ndf_ggx_numpy(alpha_sqr, cos_theta):
    """GGX NDF (numpy version, for costheta_cutoff CPU computation)."""
    cos_t = np.clip(cos_theta, 0.0, 1.0)
    d = (cos_t * alpha_sqr - cos_t) * cos_t + 1.0
    return alpha_sqr / (d * d * np.pi)


# ---- cuda_header: device functions + __global__ kernel ----

_SPECULAR_CUDA_HEADER = """
#include <cuda_runtime.h>
// Phase 37: CUDA 12.6 compat
#ifdef cudaMemcpy
#undef cudaMemcpy
#endif

__device__ float ndf_ggx_cuda(float alpha_sqr, float cos_theta) {
    float ct = fmaxf(cos_theta, 0.0f);
    float d = (ct * alpha_sqr - ct) * ct + 1.0f;
    return alpha_sqr / (d * d * 3.141592653589793f);
}

__device__ void cube_to_dir_cuda(int s, float x, float y, int N, float* dir) {
    float fx = 2.0f * ((x + 0.5f) / (float)N) - 1.0f;
    float fy = 2.0f * ((y + 0.5f) / (float)N) - 1.0f;
    float rx, ry, rz;
    switch(s) {
        case 0: rx = 1.0f;  ry = -fy;  rz = -fx; break;
        case 1: rx = -1.0f; ry = -fy;  rz = fx;  break;
        case 2: rx = fx;    ry = 1.0f; rz = fy;  break;
        case 3: rx = fx;    ry = -1.0f;rz = -fy; break;
        case 4: rx = fx;    ry = -fy;  rz = 1.0f; break;
        case 5: rx = -fx;   ry = -fy;  rz = -1.0f;break;
    }
    float norm = sqrtf(rx*rx + ry*ry + rz*rz);
    dir[0] = rx / norm; dir[1] = ry / norm; dir[2] = rz / norm;
}

__device__ float pixel_area_cuda(float x, float y, int N) {
    if (N <= 1) return 1.0f;
    float H = (float)(N / 2);
    x = fabsf(x - H);
    y = fabsf(y - H);
    float dx = atanf((x + 1.0f) / H) - atanf(x / H);
    float dy = atanf((y + 1.0f) / H) - atanf(y / H);
    return dx * dy;
}

__global__ void specular_filter_kernel(int res, float roughness, float costheta_cutoff,
    const float* __restrict__ cubemap,
    const float* __restrict__ areas,
    float* __restrict__ output)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = 6 * res * res;
    if (tid >= total_pixels) return;

    int pz = tid / (res * res);
    int remain = tid % (res * res);
    int py = remain / res;
    int px = remain % res;

    float V[3];
    cube_to_dir_cuda(pz, (float)px, (float)py, res, V);

    float alpha = roughness * roughness;
    float alpha_sqr = alpha * alpha;

    float col_r = 0.0f, col_g = 0.0f, col_b = 0.0f, wsum = 0.0f;

    for (int s = 0; s < 6; s++) {
        for (int y = 0; y < res; y++) {
            for (int x = 0; x < res; x++) {
                float L[3];
                cube_to_dir_cuda(s, (float)x, (float)y, res, L);
                float dot_L_V = L[0]*V[0] + L[1]*V[1] + L[2]*V[2];
                if (dot_L_V < costheta_cutoff) continue;

                float hx = L[0] + V[0], hy = L[1] + V[1], hz = L[2] + V[2];
                float h_norm = 1.0f / sqrtf(hx*hx + hy*hy + hz*hz);
                hx *= h_norm; hy *= h_norm; hz *= h_norm;

                float VNR_dot_H = V[0]*hx + V[1]*hy + V[2]*hz;
                if (VNR_dot_H < 0.0f) VNR_dot_H = 0.0f;

                float area = areas[y * res + x];
                float w = dot_L_V * ndf_ggx_cuda(alpha_sqr, VNR_dot_H) * area / 4.0f;

                int src_idx = (s * res * res + y * res + x) * 3;
                col_r += cubemap[src_idx + 0] * w;
                col_g += cubemap[src_idx + 1] * w;
                col_b += cubemap[src_idx + 2] * w;
                wsum += w;
            }
        }
    }

    int out_idx = tid * 3;
    output[out_idx + 0] = col_r / (wsum + 1e-10f);
    output[out_idx + 1] = col_g / (wsum + 1e-10f);
    output[out_idx + 2] = col_b / (wsum + 1e-10f);
}
"""

# ---- cuda_src: host-side kernel launch code ----

_SPECULAR_FILTER_SRC = """
@alias(cubemap, in0)
@alias(areas, in1)
int res = data["res"];
float roughness = data["roughness"];
float costheta_cutoff = data["costheta"];
int total_pixels = 6 * res * res;
int block = 256;
int grid = (total_pixels + block - 1) / block;
cudaMemset(out_p, 0, total_pixels * 3 * sizeof(float));
specular_filter_kernel<<<grid, block>>>(res, roughness, costheta_cutoff,
    cubemap_p, areas_p, out_p);
"""


def specular_cubemap(cubemap, roughness, cutoff=0.99):
    """Compute specular GGX-filtered cubemap via CUDA jt.code (brute-force, full scan)."""
    assert cubemap.shape[0] == 6 and cubemap.shape[1] == cubemap.shape[2], \
        f"Bad shape for cubemap tensor: {cubemap.shape}"

    res = cubemap.shape[1]
    total_pixels = 6 * res * res

    # Compute costheta_cutoff via numerical integration (CPU, ~0.1s)
    n_samples = 1000000
    costheta_samples = np.cos(np.linspace(0, np.pi / 2.0, n_samples))
    D = np.cumsum(_ndf_ggx_numpy(np.float64(roughness) ** 4, costheta_samples))
    idx_val = np.argmax(D >= D[-1] * cutoff)
    costheta_cutoff = float(costheta_samples[idx_val])

    # Precompute areas (cached on CPU)
    areas_np = _get_cubemap_areas(res).reshape(-1)
    areas_jt = jt.array(areas_np, dtype=jt.float32)

    cm_flat = cubemap.reshape(-1, 3)

    # Single-pass brute-force filter (GPU)
    output_flat = jt.code([total_pixels * 3], "float32",
        [cm_flat, areas_jt],
        data={'res': res, 'roughness': roughness, 'costheta': costheta_cutoff},
        cuda_src=_SPECULAR_FILTER_SRC,
        cuda_header=_SPECULAR_CUDA_HEADER)

    return output_flat.reshape(6, res, res, 3)


# ----------------------------------------------------------------------------
# Fresnel Schlick (internal helper — use bsdf_fresnel_shlick from bsdf.py)
# ----------------------------------------------------------------------------

def _fresnel_shlick(f0, f90, cosTheta):
    return bsdf_fresnel_shlick(f0, f90, cosTheta)


def _ndf_ggx(alphaSqr, cosTheta):
    return bsdf_ndf_ggx(alphaSqr, cosTheta)


def _lambda_ggx(alphaSqr, cosTheta):
    return bsdf_lambda_ggx(alphaSqr, cosTheta)


def _masking_smith(alphaSqr, cosThetaI, cosThetaO):
    return bsdf_masking_smith_ggx_correlated(alphaSqr, cosThetaI, cosThetaO)


# ----------------------------------------------------------------------------
# Shading normal setup
# ----------------------------------------------------------------------------

def prepare_shading_normal(pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm,
                           two_sided_shading=True, opengl=True):
    if perturbed_nrm is None:
        perturbed_nrm = jt.array([0, 0, 1], dtype=jt.float32)[None, None, None, ...]
    return bsdf_prepare_shading_normal(pos, view_pos, perturbed_nrm, smooth_nrm,
                                       smooth_tng, geom_nrm, two_sided_shading, opengl)


# ----------------------------------------------------------------------------
# BSDF functions (delegate to bsdf.py pure Python implementations)
# ----------------------------------------------------------------------------

def lambert(nrm, wi):
    return bsdf_lambert(nrm, wi)


def frostbite_diffuse(nrm, wi, wo, linearRoughness):
    return bsdf_frostbite(nrm, wi, wo, linearRoughness)


def pbr_specular(col, nrm, wo, wi, alpha, min_roughness=0.08):
    return bsdf_pbr_specular(col, nrm, wo, wi, alpha, min_roughness)


def pbr_bsdf(kd, arm, pos, nrm, view_pos, light_pos, min_roughness=0.08, bsdf="lambert"):
    BSDF = 0 if bsdf == 'lambert' else 1
    return bsdf_pbr(kd, arm, pos, nrm, view_pos, light_pos, min_roughness, BSDF)


# ----------------------------------------------------------------------------
# Image loss function (delegate to loss.py)
# ----------------------------------------------------------------------------

def image_loss(img, target, loss='l1', tonemapper='none'):
    return image_loss_fn(img, target, loss, tonemapper)


# ----------------------------------------------------------------------------
# Transform points/vectors (pure Jittor)
# ----------------------------------------------------------------------------

def xfm_points(points, matrix):
    return jt.matmul(
        F.pad(points, pad=(0, 1), mode='constant', value=1.0),
        jt.transpose(matrix, 1, 2)
    )


def xfm_vectors(vectors, matrix):
    return jt.matmul(
        F.pad(vectors, pad=(0, 1), mode='constant', value=0.0),
        jt.transpose(matrix, 1, 2)
    )
