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

import os
import random
import json
import numpy as np
import jittor as jt
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks, storePly
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import math
import jittor.nn as F


class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], is_pbr= False, ply_path=None, logger=None, skip_octree=False):
        """
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.resolution_scales = resolution_scales

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
                
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        print(args.source_path)
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            print("Found sparse file, assuming CPLMAP data set!")
            print(args.source_path)
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args.ds)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            if "Synthetic4Relight" in args.source_path:
                print("Found transforms_train.json file, assuming Synthetic4Relight data set!")
                scene_info = sceneLoadTypeCallbacks["Synthetic4Relight"](args.source_path, args.white_background, args.eval)
            elif "TensoIRSynthetic" in args.source_path:
                print("Found transforms_train.json file, assuming TensorIRSynthetic data set!")
                scene_info = sceneLoadTypeCallbacks["TensoIRSynthetic"](args.source_path, args.white_background, args.eval)
            else:
                print("Found transforms_train.json file, assuming Blender data set!")
                scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.random_background, args.white_background,  args.eval, ply_path=ply_path)
        else:
            assert False, "Could not recognize scene type!"

        self.gaussians.set_appearance(len(scene_info.train_cameras))
      
        
        if not self.loaded_iter:
            points = self.save_ply(scene_info.point_cloud, args.ratio, os.path.join(self.model_path, "input.ply"))
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)
            jt.sync()

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in self.resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            jt.sync()
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
            jt.sync()

        if self.loaded_iter:
            self.gaussians.load_ply_sparse_gaussian(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
            self.gaussians.load_mlp_checkpoints(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter)))
            print("Load Voxel Size: ", self.gaussians.voxel_size)
            print("Load Standard Dist: ", self.gaussians.standard_dist)
        elif not skip_octree:
            if logger is not None:
                if args.random_background:
                    logger.info("Using random background")
                elif args.white_background:
                    logger.info("Using white background")
                else:
                    logger.info("Using black background")
            # Workaround: jt.array(numpy) doesn't fully copy large arrays to GPU.
            # Keep as numpy throughout initialization, convert only at model param creation.
            points = np.unique(points.numpy() if isinstance(points, jt.Var) else points, axis=0)
            # Optional: subsample points for controlled anchor count (max_points > 0)
            do_subsample = hasattr(args, 'max_points') and args.max_points > 0
            if do_subsample and len(points) > args.max_points:
                rng = np.random.RandomState(42)
                idx = rng.choice(len(points), args.max_points, replace=False)
                points = points[idx]
                if logger: logger.info(f"Subsampled points: {len(points)} (max_points={args.max_points})")
            # Always compute correct LOD levels from camera distances.
            # Phase 41 fix: removed the levels=3/init_level=1 hack that truncated
            # the octree and made most anchors permanently invisible.
            self.gaussians.set_level(points, self.train_cameras, self.resolution_scales,
                                     args.dist_ratio, args.init_level, args.levels)
            self.gaussians.create_from_pcd(points, self.cameras_extent, logger)
            # create_from_pcd → weed_out() already initializes numpy shadows
            # (_anchor_np, _level_np, _extra_level_np, _anchor_mask, standard_dist)

    def save_ply(self, pcd, ratio, path):
        pts_src = np.ascontiguousarray(pcd.points[::ratio])
        colors_src = np.ascontiguousarray(pcd.colors[::ratio])
        storePly(path, pts_src, colors_src)  # use numpy directly, no Jittor
        return pts_src  # return numpy (jt.array(numpy) doesn't fully copy large arrays)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        try:
            self.gaussians.save_mlp_checkpoints(point_cloud_path)
        except Exception as e:
            print(f"[INFO] save_mlp_checkpoints skipped: {e}")

    def getTrainCameras(self):
        all_cams = []   
        for scale in self.resolution_scales:
            all_cams.extend(self.train_cameras[scale])
        return all_cams

    def getTestCameras(self):
        all_cams = []   
        for scale in self.resolution_scales:
            all_cams.extend(self.test_cameras[scale])
        return all_cams

    def get_canonical_rays(self, scale= 1.0) -> jt.Var:
        # NOTE: some datasets do not share the same intrinsic (e.g. DTU)
        # get reference camera
        ref_camera = self.train_cameras[scale][0]
        # TODO: inject intrinsic
        H, W = ref_camera.image_height, ref_camera.image_width
        cen_x = W / 2
        cen_y = H / 2
        tan_fovx = math.tan(ref_camera.FoVx * 0.5)
        tan_fovy = math.tan(ref_camera.FoVy * 0.5)
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
    