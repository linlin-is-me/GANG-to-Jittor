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

import jittor as jt
from jittor import nn  # TODO: verify each import is valid in Jittor
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix,fov2focal

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image,normal,albedo,roughness,
                  metal,irradiance, gt_alpha_mask,
                 image_name, resolution_scale, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()
        

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.resolution_scale = resolution_scale

        self.data_device = data_device
        
        self.original_image = image.clamp(0.0, 1.0)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if normal is not None:
            self.normal = normal
            self.albedo = albedo
            self.roughness = roughness
            self.metal = metal
            self.irradiance = irradiance 
        else:
            self.normal = self.albedo = self.roughness = self.metal = self.irradiance = None
    
 
        if gt_alpha_mask is not None:
            self.mask = gt_alpha_mask.clamp(0.0, 1.0)
            self.original_image *= self.mask
            if normal is not None:
                self.normal *= self.mask
                self.albedo *= self.mask
                self.roughness *= self.mask
                self.metal *= self.mask
                self.irradiance *= self.mask

        else:
            self.mask = None
            self.original_image *= jt.ones((1, self.image_height, self.image_width))
            if normal is not None:
                self.normal *= jt.ones((1, self.image_height, self.image_width))
                self.albedo *= jt.ones((1, self.image_height, self.image_width))
                self.roughness *= jt.ones((1, self.image_height, self.image_width))
                self.metal *= jt.ones((1, self.image_height, self.image_width))
                self.irradiance *= jt.ones((1, self.image_height, self.image_width))

        prcppoint = np.array([0.5, 0.5])
        self.prcppoint = jt.array(prcppoint).float32()  # 

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = jt.array(getWorld2View2(R, T, trans, scale)).transpose(0, 1)
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0) @ self.projection_matrix.unsqueeze(0)).squeeze(0)
        self.camera_center = jt.linalg.inv(self.world_view_transform)[3, :3]
        
    def get_calib_matrix_nerf(self):
        focal = fov2focal(self.FoVx, self.image_width)  # original focal length
        intrinsic_matrix = jt.array([[focal, 0, self.image_width / 2], [0, focal, self.image_height / 2], [0, 0, 1]]).float()
        extrinsic_matrix = self.world_view_transform.transpose(0,1).contiguous() # cam2world
        return intrinsic_matrix, extrinsic_matrix


class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = jt.linalg.inv(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

