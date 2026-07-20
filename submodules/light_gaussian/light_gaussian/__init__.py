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

from typing import NamedTuple
from jittor import nn
import jittor as jt
# Import from sibling module in the same package (light_gaussian/)
# Import from sibling file in the parent light_gaussian/ package
import sys, os
_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)
import rasterize_points_jt as _rasterize_points_jt
from rasterize_points_jt import (
    RasterizeGaussiansCUDA, RasterizeGaussiansBackwardCUDA, RasterizeGaussiansFilterCUDA,
    markVisible, depthToNormal, SurfaceAlignCUDA, SurfaceAlignBackwardCUDA,
)

# Phase 22: Global cache for CUDA backward gradients (bypass Jittor autograd crash)
_GRAD_CACHE = {}

class _surface_align(jt.Function):

    def save_for_backward(self, *args):
        self.saved_tensors = args

    def execute(self, anchor, offsets_all, rotation, knn_index):
        """anchor (N,3), offsets_all (N*K,3), rotation (N*K,4), knn_index (N,K)"""
        K = knn_index.shape[1]
        from ..rasterize_points_jt import repeat_cuda
        with jt.no_grad():
            repeat_anchor_all = repeat_cuda(anchor, K)       # (N,3) -> (N*K,3)
        xyz_all = repeat_anchor_all + offsets_all            # (N*K,3)

        loss_d, loss_normal, binning_buffer, mean_d = SurfaceAlignCUDA(xyz_all, rotation, knn_index)
        self.save_for_backward(anchor, offsets_all, rotation, binning_buffer, knn_index, mean_d)
        return loss_d, loss_normal

    def grad(self, grad_out_loss_d, grad_out_loss_normal):
        anchor, offsets_all, rotation, binning_buffer, knn_index, mean_d = self.saved_tensors
        K = knn_index.shape[1]

        if grad_out_loss_d is None:
            grad_out_loss_d = jt.zeros_like(mean_d)
        if grad_out_loss_normal is None:
            grad_out_loss_normal = jt.zeros_like(mean_d)

        from ..rasterize_points_jt import repeat_cuda, repeat_sum_bwd_cuda
        with jt.no_grad():
            repeat_anchor_all = repeat_cuda(anchor, K)
        xyz_all = repeat_anchor_all + offsets_all

        grad_xyz, grad_rotation = SurfaceAlignBackwardCUDA(
            xyz_all, rotation, mean_d, knn_index,
            grad_out_loss_d, grad_out_loss_normal)

        grad_anchor = repeat_sum_bwd_cuda(grad_xyz, K)     # (N*K,3) -> (N,3)

        # NOTE: do NOT set self.saved_tensors = None — retain_graph=True
        # requires them for the second backward pass (optimizer.backward)
        return grad_anchor, grad_xyz, grad_rotation, None


class SurfaceAlign(nn.Module):
    def __init__(self):
        super().__init__()
        self.alignFunc = _surface_align()

    def execute(self, anchor, offsets_all, rotation, knn_index):
        return self.alignFunc(anchor, offsets_all, rotation, knn_index)


def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.clone() if isinstance(item, jt.Var) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    norm3Ds_precomp,
    extra_attrs,
    raster_settings,
):
    num_contrib, color, depth, opacity, norm, alpha, radii, extra = _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        norm3Ds_precomp,
        extra_attrs,
        raster_settings,
    )

    norm = jt.normalize(norm, p=2, dim=0)

    focal_x = raster_settings.image_width / (2.0 * raster_settings.tanfovx)
    focal_y = raster_settings.image_height / (2.0 * raster_settings.tanfovy)
    # TODO Phase 3: replace kornia.filters.median_blur with Jittor implementation
    depth_filter = depth  # fallback: skip median blur until Phase 3
    normal_from_depth = depthToNormal(
        depth_filter.squeeze(0) if depth_filter.ndim == 3 else depth_filter,
        raster_settings.viewmatrix,
        focal_x,
        focal_y,
    )
    return num_contrib, color, depth, opacity, norm, normal_from_depth, alpha, radii, extra

class _RasterizeGaussians(jt.Function):

    def save_for_backward(self, *args):
        self.saved_tensors = args

    def execute(
        self,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        norm3Ds_precomp,
        extra_attrs,
        raster_settings,
    ):
        assert extra_attrs.shape[0] == 0 or extra_attrs.shape[1] <= 34

        if raster_settings.debug:
            try:
                num_rendered, num_contrib, color, depth, opacity, norm, alpha, extra, radii, geomBuffer, binningBuffer, imgBuffer = RasterizeGaussiansCUDA(
                    raster_settings.bg, means3D, colors_precomp, opacities, scales, rotations,
                    raster_settings.scale_modifier, cov3Ds_precomp, norm3Ds_precomp, extra_attrs,
                    raster_settings.viewmatrix, raster_settings.projmatrix,
                    raster_settings.tanfovx, raster_settings.tanfovy,
                    raster_settings.image_height, raster_settings.image_width,
                    sh, raster_settings.sh_degree, raster_settings.campos,
                    raster_settings.prefiltered, raster_settings.debug)
            except Exception as ex:
                jt.save([means3D, colors_precomp, opacities, scales, rotations, extra_attrs,
                         raster_settings.viewmatrix, raster_settings.projmatrix], "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, num_contrib, color, depth, opacity, norm, alpha, extra, radii, geomBuffer, binningBuffer, imgBuffer = RasterizeGaussiansCUDA(
                raster_settings.bg, means3D, colors_precomp, opacities, scales, rotations,
                raster_settings.scale_modifier, cov3Ds_precomp, norm3Ds_precomp, extra_attrs,
                raster_settings.viewmatrix, raster_settings.projmatrix,
                raster_settings.tanfovx, raster_settings.tanfovy,
                raster_settings.image_height, raster_settings.image_width,
                sh, raster_settings.sh_degree, raster_settings.campos,
                raster_settings.prefiltered, raster_settings.debug)

        self.raster_settings = raster_settings
        self.num_rendered = num_rendered
        self.save_for_backward(colors_precomp, means3D, scales, rotations,
                               cov3Ds_precomp, norm3Ds_precomp, radii, extra_attrs,
                               sh, geomBuffer, binningBuffer, imgBuffer, alpha)
        return num_contrib, color, depth, opacity, norm, alpha, radii, extra

    def grad(self, grad_out_contrib, grad_out_color, grad_out_depth, grad_out_opacity, grad_out_norm, grad_out_alpha, _, grad_out_extra):
        num_rendered = self.num_rendered
        raster_settings = self.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, norm3Ds_precomp, \
            radii, extra_attrs, sh, geomBuffer, binningBuffer, imgBuffer, alpha = self.saved_tensors

        if grad_out_color is None:
            grad_out_color = jt.zeros((3, grad_out_depth.shape[1], grad_out_depth.shape[2]))
        if grad_out_depth is None:
            grad_out_depth = jt.zeros((1, grad_out_color.shape[1], grad_out_color.shape[2]))
        if grad_out_norm is None:
            grad_out_norm = jt.zeros((3, grad_out_color.shape[1], grad_out_color.shape[2]))
        if grad_out_alpha is None:
            grad_out_alpha = jt.zeros((1, grad_out_color.shape[1], grad_out_color.shape[2]))
        if grad_out_extra is None:
            if extra_attrs.shape[0] != 0:
                grad_out_extra = jt.zeros((extra_attrs.shape[1], grad_out_color.shape[1], grad_out_color.shape[2]))
            else:
                grad_out_extra = jt.zeros([1])

        if raster_settings.debug:
            try:
                grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_norm3Ds_precomp, grad_sh, grad_scales, grad_rotations, grad_extra_attrs = RasterizeGaussiansBackwardCUDA(
                    raster_settings.bg, means3D, radii, colors_precomp, scales, rotations, extra_attrs,
                    raster_settings.scale_modifier, cov3Ds_precomp, norm3Ds_precomp,
                    raster_settings.viewmatrix, raster_settings.projmatrix,
                    raster_settings.tanfovx, raster_settings.tanfovy,
                    grad_out_color, grad_out_depth, grad_out_norm, grad_out_alpha, grad_out_extra,
                    sh, raster_settings.sh_degree, raster_settings.campos,
                    geomBuffer, num_rendered, binningBuffer, imgBuffer, alpha, raster_settings.debug)
            except Exception as ex:
                jt.save([raster_settings.bg, means3D, radii, colors_precomp, scales, rotations, extra_attrs,
                         raster_settings.viewmatrix, raster_settings.projmatrix,
                         grad_out_color, grad_out_depth, grad_out_norm, grad_out_alpha, grad_out_extra], "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
            grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_norm3Ds_precomp, grad_sh, grad_scales, grad_rotations, grad_extra_attrs = RasterizeGaussiansBackwardCUDA(
                raster_settings.bg, means3D, radii, colors_precomp, scales, rotations, extra_attrs,
                raster_settings.scale_modifier, cov3Ds_precomp, norm3Ds_precomp,
                raster_settings.viewmatrix, raster_settings.projmatrix,
                raster_settings.tanfovx, raster_settings.tanfovy,
                grad_out_color, grad_out_depth, grad_out_norm, grad_out_alpha, grad_out_extra,
                sh, raster_settings.sh_degree, raster_settings.campos,
                geomBuffer, num_rendered, binningBuffer, imgBuffer, alpha, raster_settings.debug)

        # NOTE: do NOT set self.saved_tensors = None — retain_graph=True
        # requires them for the second backward pass (optimizer.backward)
        return (
            grad_means3D,         # means3D
            grad_means2D,         # means2D
            grad_sh,              # sh
            grad_colors_precomp,  # colors_precomp
            grad_opacities,       # opacities
            grad_scales,          # scales
            grad_rotations,       # rotations
            grad_cov3Ds_precomp,  # cov3Ds_precomp
            grad_norm3Ds_precomp, # norm3Ds_precomp
            grad_extra_attrs,     # extra_attrs
            None                  # raster_settings
        )

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int
    tanfovx : float
    tanfovy : float
    bg : jt.Var
    scale_modifier : float
    viewmatrix : jt.Var
    projmatrix : jt.Var
    sh_degree : int
    campos : jt.Var
    prefiltered : bool
    debug : bool

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings
        # Persistent Function instance: Jittor 1.3.11's tape_together may lose
        # the Python instance reference during C++→Python callback, causing
        # "saved_tensors missing" on a fresh instance. Persistent instance avoids GC.
        self._rasterize_func = _RasterizeGaussians()

    def markVisible(self, positions):
        with jt.no_grad():
            raster_settings = self.raster_settings
            visible = markVisible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
        return visible

    def execute(self, means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3Ds_precomp = None, norm3Ds_precomp=None, extra_attrs=None):

        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')

        if ((scales is None or rotations is None) and cov3Ds_precomp is None) or ((scales is not None or rotations is not None) and cov3Ds_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')

        if shs is None:
            shs = jt.array([])
        if colors_precomp is None:
            colors_precomp = jt.array([])

        if scales is None:
            raise ValueError('To support norm and depth prediction, scales == None is not allowed')
        if rotations is None:
            raise ValueError('To support norm and depth prediction, rotations == None is not allowed')
        if cov3Ds_precomp is None:
            cov3Ds_precomp = jt.array([])
        if norm3Ds_precomp is None:
            norm3Ds_precomp = jt.array([])
        if extra_attrs is None:
            extra_attrs = jt.array([])
        # Store rasterizer inputs for manual gradient computation (Phase 22)
        self._last_scales_input = scales
        self._last_rotations_input = rotations
        self._last_means3D_input = means3D
        self._last_opacities_input = opacities
        # Manual instance creation: keep reference for saved_tensors access
        self._last_rasterize_func = _RasterizeGaussians()
        num_contrib, color, depth, opacity, norm, alpha, radii, extra = \
            self._last_rasterize_func(
                means3D,
                means2D,
                shs,
                colors_precomp,
                opacities,
                scales,
                rotations,
                cov3Ds_precomp,
                norm3Ds_precomp,
                extra_attrs,
                raster_settings,
            )

        # Post-processing (inlined from rasterize_gaussians)
        norm = jt.normalize(norm, p=2, dim=0)
        focal_x = raster_settings.image_width / (2.0 * raster_settings.tanfovx)
        focal_y = raster_settings.image_height / (2.0 * raster_settings.tanfovy)
        depth_filter = depth
        normal_from_depth = depthToNormal(
            depth_filter.squeeze(0) if depth_filter.ndim == 3 else depth_filter,
            raster_settings.viewmatrix,
            focal_x,
            focal_y,
        )
        return num_contrib, color, depth, opacity, norm, normal_from_depth, alpha, radii, extra

    def visible_filter(self, means3D, scales = None, rotations = None, cov3D_precomp = None):

        raster_settings = self.raster_settings

        if scales is None:
            scales = jt.array([])
        if rotations is None:
            rotations = jt.array([])
        if cov3D_precomp is None:
            cov3D_precomp = jt.array([])

        with jt.no_grad():
            radii = RasterizeGaussiansFilterCUDA(
                means3D, scales, rotations,
                raster_settings.scale_modifier,
                cov3D_precomp,
                raster_settings.viewmatrix,
                raster_settings.projmatrix,
                raster_settings.tanfovx,
                raster_settings.tanfovy,
                raster_settings.image_height,
                raster_settings.image_width,
                raster_settings.prefiltered,
                raster_settings.debug)
        return radii
