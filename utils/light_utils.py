# light_gaussian imports — handle both old (C++ ext) and new (Jittor jt.code) paths
_C = None  # compiled C++ extension (lite_rasterize_gaussians) — not in Jittor path
GaussianRasterizationSettings = None
GaussianRasterizer = None

import jittor as jt
import jittor.nn as F
from utils.graphics_utils import getProjectionMatrix
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union
from copy import deepcopy

try:
    from light_gaussian import _C as _C_ext, GaussianRasterizationSettings, GaussianRasterizer
    _C = _C_ext
except (ImportError, ModuleNotFoundError, ValueError):
    # Jittor path: _C not available; GaussianRasterizationSettings/GaussianRasterizer
    # are inside light_gaussian.light_gaussian sub-package
    try:
        from light_gaussian.light_gaussian import GaussianRasterizationSettings, GaussianRasterizer
    except (ImportError, ModuleNotFoundError, ValueError):
        pass  # Will only fail if these are actually used

def get_canonical_rays(H: int, W: int, tan_fovx: float, tan_fovy: float) -> jt.Var:
    cen_x = W / 2
    cen_y = H / 2
    focal_x = W / (2.0 * tan_fovx)
    focal_y = H / (2.0 * tan_fovy)

    x, y = jt.meshgrid(
        jt.arange(W),
        jt.arange(H),
        indexing="xy",
    )
    x = x.flatten()  # [H * W]
    y = y.flatten()  # [H * W]
    camera_dirs = F.pad(
        jt.stack(
            [
                (x - cen_x + 0.5) / focal_x,
                (y - cen_y + 0.5) / focal_y,
            ],
            dim=-1,
        ),
        (0, 1),
        value=1.0,
    )  # [H * W, 3]
    # NOTE: it is not normalized
    return camera_dirs


def getWorld2ViewTorch(R: jt.Var, t: jt.Var) -> jt.Var:
    Rt = jt.zeros((4, 4))
    Rt[:3, :3] = R[:3, :3].T
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return Rt


# inverse the mapping from https://github.com/NVlabs/nvdiffrec/blob/dad3249af8ede96c7dd72c30328272117fabb710/render/light.py#L22
def get_envmap_dirs(res = [256, 512]) -> jt.Var:
    gy, gx = jt.meshgrid(
        jt.linspace(0.0, 1.0 - 1.0 / res[0], res[0]),
        jt.linspace(-1.0, 1.0 - 1.0 / res[1], res[1]),
        indexing="ij",
    )

    sintheta, costheta = jt.sin(gy * np.pi), jt.cos(gy * np.pi)
    sinphi, cosphi = jt.sin(gx * np.pi), jt.cos(gx * np.pi)

    reflvec = jt.stack((sintheta * sinphi, costheta, -sintheta * cosphi), dim=-1)  # [H, W, 3]

    return reflvec

def get_depth_cubemap(get_xyz,get_opacity,get_scaling,get_rotation,get_features, position, res = 512
):
    canonical_rays = get_canonical_rays(H=res, W=res, tan_fovx=1.0, tan_fovy=1.0)  # [HW, 3]
    norm = jt.norm(canonical_rays, p=2, dim=-1).reshape(res, res, 1)  # [H, W]

    bg_color = jt.zeros([3, res, res])
    rotations: List[jt.Var] = [
        jt.array(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ),  # lookAt(jt.array([0, 0, 0]), jt.array([-1.0, 0.0, 0.0]), jt.array([0.0, -1.0, 0.0]))  [eye, center, up]
        jt.array(
            [
                [0.0, 0.0, -1.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ),  # lookAt(jt.array([0, 0, 0]), jt.array([1.0, 0.0, 0.0]), jt.array([0.0, -1.0, 0.0]))  [eye, center, up]
        jt.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ),  # lookAt(jt.array([0, 0, 0]), jt.array([0.0, -1.0, 0.0]), jt.array([0.0, 0.0, -1.0]))  [eye, center, up]
        jt.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ),  # lookAt(jt.array([0, 0, 0]), jt.array([0.0, 1.0, 0.0]), jt.array([0.0, 0.0, 1.0]))  [eye, center, up]
        jt.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ),  # lookAt(jt.array([0, 0, 0]), jt.array([0.0, 0.0, -1.0]), jt.array([0.0, 1.0, 0.0]))  [eye, center, up]
        jt.array(
            [
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ),  # lookAt(jt.array([0, 0, 0]), jt.array([0.0, 0.0, 1.0]), jt.array([0.0, -1.0, 0.0]))  [eye, center, up]
    ]
    zfar = 100.0
    znear = 0.01
    projection_matrix = (
        getProjectionMatrix(znear=znear, zfar=zfar, fovX=np.pi * 0.5, fovY=np.pi * 0.5)
        .transpose(0, 1)
        
    )

    depth_cubemap = []
    opacity_cubemap = []
    for r_idx, rotation in enumerate(rotations):
        c2w = rotation
        c2w[:3, 3] = position
        w2c = jt.linalg.inv(c2w)
        T = w2c[:3, 3]
        R = w2c[:3, :3].T
        world_view_transform = getWorld2ViewTorch(R, T).transpose(0, 1)
        full_proj_transform = (
            world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
        ).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]

        input_args = (
            bg_color,
            # bg_colors[r_idx],
            get_xyz,
            jt.Var([]),
            get_opacity,
            get_scaling,
            get_rotation,
            jt.Var([]),
            get_features,
            camera_center,  # campos,
            world_view_transform,  # viewmatrix,
            full_proj_transform,  # projmatrix,
            1.0,  # scale_modifier
            1.0,  # tanfovx,
            1.0,  # tanfovy,
            res,  # image_height,
            res,  # image_width,
            1,
            False,  # prefiltered,
            True,  # argmax_depth, 
        )
        if _C is None:
            return None, None
        (num_rendered, rendered_image, opacity_map, radii, depth_map) = _C.lite_rasterize_gaussians(*input_args)

        # depth_cubemap.append(depth_map.permute(1, 2, 0) * norm)
        depth_cubemap.append(depth_map.permute(1, 2, 0))
        opacity_cubemap.append(opacity_map.permute(1, 2, 0))

    return jt.stack(depth_cubemap), jt.stack(opacity_cubemap)


# def get_depth_cubemap_moving(get_xyz,get_opacity,get_scaling,get_rotation,get_features,rotations, position, res = 256
# ):
#     # get canonical ray and its norm to normalize depth
#     canonical_rays = get_canonical_rays(H=res, W=res, tan_fovx=1.0, tan_fovy=1.0)  # [HW, 3]
#     norm = jt.norm(canonical_rays, p=2, dim=-1).reshape(res, res, 1)  # [H, W]

#     bg_color = jt.zeros([3, res, res])
    
#     zfar = 100.0
#     znear = 0.01
#     projection_matrix = (
#         getProjectionMatrix(znear=znear, zfar=zfar, fovX=np.pi * 0.5, fovY=np.pi * 0.5)
#         .transpose(0, 1)
#         
#     )

#     depth_cubemap = []
#     opacity_cubemap = []
#     for r_idx, rotation in enumerate(rotations):
#         print(r_idx)
#          c2w = rotations[r_idx]
#         # print(c2w.shape,position.shape,type(c2w),type(position))
#         c2w[:3, 3] = position
#         w2c = jt.linalg.inv(c2w)
#         T = w2c[:3, 3]
#         R = w2c[:3, :3].T
#         world_view_transform = getWorld2ViewTorch(R, T).transpose(0, 1)
#         full_proj_transform = (
#             world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
#         ).squeeze(0)
#         camera_center = world_view_transform.inverse()[3, :3]

#         input_args = (
#             bg_color,
#             # bg_colors[r_idx],
#             get_xyz,
#             jt.Var([]),
#             get_opacity,
#             get_scaling,
#             get_rotation,
#             jt.Var([]),
#             get_features,
#             camera_center,  # campos,
#             world_view_transform,  # viewmatrix,
#             full_proj_transform,  # projmatrix,
#             1.0,  # scale_modifier
#             1.0,  # tanfovx,
#             1.0,  # tanfovy,
#             res,  # image_height,
#             res,  # image_width,
#             1,
#             False,  # prefiltered,
#             True,  # argmax_depth, 
#         )
#         (num_rendered, rendered_image, opacity_map, radii, depth_map) = _C.lite_rasterize_gaussians(*input_args)

#         # depth_cubemap.append(depth_map.permute(1, 2, 0) * norm)
#         depth_cubemap.append(depth_map.permute(1, 2, 0))
#         opacity_cubemap.append(opacity_map.permute(1, 2, 0))

#     return jt.stack(depth_cubemap), jt.stack(opacity_cubemap)

    


def turbo_cmap(gray: np.ndarray) -> np.ndarray:
    """
    Visualize a single-channel image using matplotlib's turbo color map
    yellow is high value, blue is low
    :param gray: np.ndarray, (H, W) or (H, W, 1) unscaled
    :return: (H, W, 3) float32 in [0, 1]
    """
    colored = plt.cm.turbo(plt.Normalize()(gray.squeeze()))[..., :-1]
    return colored.astype(np.float32)



def DistributionGGX(
    normals: jt.Var,  # [H, W, 3]
    half_dirs: jt.Var,  # [H, W, 3]
    roughness: jt.Var,  # [H, W, 1]
) -> jt.Var:
    a = roughness * roughness
    a2 = a * a
    NoH = saturate_dot(normals, half_dirs)
    
    NoH2 = NoH * NoH

    nom = a2
    denom = (NoH2 * (a2 - 1.0) + 1.0)
    denom = np.pi * denom * denom + 1e-4
    # print("nom",nom.max(),nom.min())
    # print("denom",denom.max(),denom.min())
    # print("NoH2",NoH2.max(),NoH2.min())

    return nom / denom

def saturate_dot(a: jt.Var, b: jt.Var) -> jt.Var:
    return (a * b).sum(dim=-1, keepdim=True).clamp(0.0, 1.0)


def GeometrySchlickGGX(
    NoV: jt.Var, # [H, W, 1]
    roughness: jt.Var,  # [H, W, 1]
) -> jt.Var:
    r = roughness + 1.0
    k = (r * r) / 8.0
    nom = NoV
    denom = NoV * (1.0 - k) + k

    return nom / denom

def GeometrySmith(
    normals: jt.Var,  # [H, W, 3]
    view_dirs: jt.Var,  # [H, W, 3]
    light_dirs: jt.Var,  # [H, W, 3]
    roughness: jt.Var,  # [H, W, 1]
) -> jt.Var:
    NoV = saturate_dot(normals, view_dirs)
    NoL = saturate_dot(normals, light_dirs)
    ggx2 = GeometrySchlickGGX(NoV, roughness)
    ggx1 = GeometrySchlickGGX(NoL, roughness)

    return ggx1 * ggx2


def fresnelSchlick(
    HoV: jt.Var,  # [H, W, 1]
    F0: jt.Var,  # [H, W, 3]
) -> jt.Var:
    return F0 + (1.0 - F0) * jt.pow((1.0 - HoV).clamp(0.0, 1.0), 5)

def linear_to_srgb(linear: Union[np.ndarray, jt.Var]) -> Union[np.ndarray, jt.Var]:
    if isinstance(linear, jt.Var):
        """Assumes `linear` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
        eps = jt.finfo(jt.float32).eps
        srgb0 = 323 / 25 * linear
        srgb1 = (211 * jt.clamp(linear, min=eps) ** (5 / 12) - 11) / 200
        return jt.where(linear <= 0.0031308, srgb0, srgb1)
    elif isinstance(linear, np.ndarray):
        eps = np.finfo(np.float32).eps
        srgb0 = 323 / 25 * linear
        srgb1 = (211 * np.maximum(eps, linear) ** (5 / 12) - 11) / 200
        return np.where(linear <= 0.0031308, srgb0, srgb1)
    else:
        raise NotImplementedError


# https://github.com/JoeyDeVries/LearnOpenGL/blob/master/src/6.pbr/2.2.1.ibl_specular/2.2.1.pbr.fs
def light_pbr_shading(
    light_position: jt.Var,  # [3]
    light_intensity: jt.Var,  # [3]
    points: jt.Var,  # [H, W, 3]
    normals: jt.Var,  # [H, W, 3]
    view_dirs: jt.Var,  # [H, W, 3]
    albedo: jt.Var,  # [H, W, 3]
    roughness: jt.Var,  # [H, W, 1]
    mask: jt.Var,  # [H, W, 1]
    linear: bool = False,
    metallic: Optional[jt.Var] = None,
    shadow: Optional[jt.Var] = None,
    background: Optional[jt.Var] = None,
) -> Dict:
    if background is None:
        background = jt.zeros_like(normals)  # [H, W, 3]

    # preapre
    light_dirs = jt.normalize(light_position - points, p=2, dim=-1)  # [H, W, 3]
    half_dirs = (light_dirs + view_dirs) / 2.0  # [H, W, 3]
    distance = jt.norm(light_position - points, p=2, dim=-1, keepdim=True)  # [H, W, 1]
    attenuation = 1.0 / jt.pow(distance, 2)  # [H, W, 1]
    radiance = light_intensity * attenuation  # [H, W, 3]

    if metallic is None:
        F0 = jt.ones_like(albedo) * 0.04  # [H, W, 3]
    else:
        F0 = (1.0 - metallic) * 0.04 + albedo * metallic  # [H, W, 3]

    # Cook-Torrance BRDF
    NoV = saturate_dot(normals, view_dirs)  # [H, W, 1]
    NoL = saturate_dot(normals, light_dirs)  # [H, W, 1]
    HoV = saturate_dot(half_dirs, view_dirs)  # [H, W, 1]
    NDF = DistributionGGX(normals=normals, half_dirs=half_dirs, roughness=roughness)  # [H, W, 1]
    G = GeometrySmith(normals=normals, view_dirs=view_dirs, light_dirs=light_dirs, roughness=roughness)  # [H, W, 1]
    fresnel = fresnelSchlick(HoV=HoV, F0=F0)  # [H, W, 3]

    numerator = NDF * G * fresnel  # [H, W, 3]
    denominator = 4.0 * NoV * NoL + 1e-4  # [H, W, 1]
    specular = numerator / denominator + 1e-4  # [H, W, 3]

    kd = 1.0 - fresnel  # [H, W, 3]
    if metallic is not None:
        kd *= (1.0 - metallic)
    
    render_rgb = (kd * albedo / np.pi + specular) * radiance # * NoL

    render_rgb = jt.where(mask, render_rgb, background)

    if shadow is not None:
        render_rgb = jt.where(shadow == 0.0, render_rgb*0.2, render_rgb)

    # if linear:
    render_rgb = linear_to_srgb(render_rgb.squeeze())

    results = {}
    results.update(
        {
            "render_rgb": render_rgb,
        }
    )

    return results