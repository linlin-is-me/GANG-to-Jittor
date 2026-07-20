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

import time
from datetime import timedelta
import jittor as jt
from functools import reduce
import numpy as np
# torch_scatter.scatter_max replaced with native torch.scatter_reduce (Windows compat)
from utils.general_utils import inverse_sigmoid, get_expon_lr_func
from utils.jt_safe import path_log
from jittor import nn  # TODO: verify each import is valid in Jittor
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
# lazy import: from gaussian_renderer.simple_knn_jt import distCUDA2  (moved inside create_from_pcd to avoid circular import)
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation,build_rotation
from scene.embedding import Embedding
import math
from utils.graphics_utils import fibonacci_sphere_sampling,get_minimum_axis,flip_align_view
try:
    import open3d as o3d
except ImportError:
    o3d = None
from utils.general_utils import quaternion2rotmat
# from submodules.permuto_sdf.permuto_sdf_py.models.models import SDF
# from submodules.permuto_sdf.permuto_sdf_py.utils.common_utils import create_bb_for_dataset
# from SDF.network import SDF
from SDF.dpsr import DPSR
import sys

# Jittor-compatible quantile (replaces torch.quantile, which has no Jittor equivalent)
# jt.sort() returns (sorted_values, indices) — Jittor 1.3.11 misc.py:145
def jt_quantile(x, q):
    """Quantile for 1-D tensor using pure Jittor jt.sort() (GPU, no numpy).

    Jittor 1.3.11: jt.sort() wraps jt.argsort() which uses thrust::sort on CUDA.
    """
    sorted_x, _ = jt.sort(x.reshape(-1))  # jt.sort returns (value, index)
    n = sorted_x.shape[0]
    idx = q * (n - 1)
    lo = int(jt.floor(idx).numpy())
    hi = min(lo + 1, n - 1)
    frac = idx - float(lo)
    return float(sorted_x[lo].numpy()) + frac * (float(sorted_x[hi].numpy()) - float(sorted_x[lo].numpy()))


def _jt_bool_to_np_indices(mask):
    """Convert Jittor boolean mask to integer indices via GPU jt.nonzero().

    Jittor 1.3.11: jt.nonzero() uses where_op.cc CUDA kernel (warp/block/CUB).
    Returns jt.Var [M] on GPU — no CPU round-trip, preserves grad capability.
    """
    if mask is None:
        return None
    idx = jt.nonzero(mask)  # GPU — where_op.cc CUDA kernel
    if idx.ndim > 1:
        idx = idx.squeeze(1)
    return idx  # jt.Var [M]


def _jt_safe_unique_2d(coords):
    """Compute unique rows via jt.unique(dim=0). Jittor 1.3.11 uses thrust::sort+unique."""
    return jt.unique(coords, return_inverse=True, dim=0)


# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../submodules/permuto_sdf/permuto_sdf_py/models')))
# from models import SDF

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../submodules/permuto_sdf/permuto_sdf_py/utils')))
# from common_utils import create_bb_for_dataset


def sample_incident_rays(normals, is_training=False, sample_num=24):
    if is_training:
        incident_dirs, incident_areas = fibonacci_sphere_sampling(
            normals, sample_num, random_rotate=True)
    else:
        incident_dirs, incident_areas = fibonacci_sphere_sampling(
            normals, sample_num, random_rotate=False)

    return incident_dirs, incident_areas  # [N, S, 3], [N, S, 1]


class ClipLayer(nn.Module):
    def __init__(self, min_val, max_val):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def execute(self, x):
        return x.maximum(self.min_val).minimum(self.max_val)
    
class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = jt.exp
        self.scaling_inverse_activation = jt.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = jt.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = jt.normalize


    def __init__(self, 
                 feat_dim: int=32, 
                 n_offsets: int=5, 
                 fork: int=2,
                 use_feat_bank : bool = False,
                 appearance_dim : int = 32,
                 add_opacity_dist : bool = False,
                 add_cov_dist : bool = False,
                 add_color_dist : bool = False,
                 add_level: bool = False,
                 visible_threshold: float = -1,
                 dist2level: str = 'round',
                 base_layer: int = 10,
                 progressive: bool = True,
                 extend: float = 1.1,
                 is_pbr: bool = False,
                 normal_detal: bool=False,
                 with_matallic:bool=True,
                 grid_resolution:int=128
                 ):
        self.normal_detal = normal_detal
        self.with_matallic = with_matallic
        self.feat_dim = feat_dim
        self.view_dim = 3
        self.n_offsets = n_offsets
        self.fork = fork
        self.use_feat_bank = use_feat_bank
        self.is_pbr = is_pbr

        self.centroid = None
        
        self.is_sdf = True

        self.appearance_dim = appearance_dim
        self.embedding_appearance = None
        self.add_opacity_dist = add_opacity_dist
        self.add_cov_dist = add_cov_dist
        self.add_color_dist = add_color_dist
        self.add_level = add_level
        self.progressive = progressive

        # SDF
        # self.SDF = SDF(in_channels=3, geom_feat_size_out=32, nr_iters_for_c2f=10000*1.0)
        # self.dpsr = DPSR(res=(grid_resolution,grid_resolution,grid_resolution),sig=2)
        self.dpsr = DPSR(res=(256,256,256),sig=2)

        # Octree
        self.sub_pos_offsets = jt.array([[i % fork, (i // fork) % fork, i // (fork * fork)] for i in range(fork**3)]).float()
        self.extend = extend
        self.visible_threshold = visible_threshold
        self.dist2level = dist2level
        self.base_layer = base_layer
        
        self.start_step = 0
        self.end_step = 0

        self._anchor = jt.empty(0)
        self._level = jt.empty(0)
        self._offset = jt.empty(0)
        self._anchor_feat = jt.empty(0)
        self.opacity_accum = jt.empty(0)
        self._scaling = jt.empty(0)
        self._rotation = jt.empty(0)
        self._opacity = jt.empty(0)
        
        self.offset_gradient_accum = jt.empty(0)
        self.offset_denom = jt.empty(0)

        self.anchor_demon = jt.empty(0)
                
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        self.opacity_dist_dim = 1 if self.add_opacity_dist else 0
        self.cov_dist_dim = 1 if self.add_cov_dist else 0
        self.color_dist_dim = 1 if self.add_color_dist else 0
        self.level_dim = 1 if self.add_level else 0

        self.albedo = jt.empty(0)
        self.roughness = jt.empty(0)
        self.matallic = jt.empty(0)

        if self.use_feat_bank:
            self.mlp_feature_bank = nn.Sequential(
                    nn.Linear(self.view_dim+self.level_dim, self.feat_dim),
                    nn.ReLU(),
                    nn.Linear(self.feat_dim, 3),
                    nn.Softmax(dim=1)
                )
        self.mlp_opacity = nn.Sequential(
                nn.Linear(self.feat_dim+self.view_dim+self.opacity_dist_dim+self.level_dim, self.feat_dim),
                nn.ReLU(),
                nn.Linear(self.feat_dim, self.n_offsets),
                nn.Tanh()      # Phase 42: Tanh matches PyTorch GANG (Sigmoid was stuck at 0.5 → no diversity)
            )
        self.mlp_roughness = nn.Sequential(
                nn.Linear(self.feat_dim+self.view_dim+self.opacity_dist_dim+self.level_dim, self.feat_dim),
                nn.ReLU(),
                nn.Linear(self.feat_dim, self.n_offsets),
                nn.Sigmoid()
            )

        self.mlp_matallic = nn.Sequential(
                nn.Linear(self.feat_dim+self.view_dim+self.opacity_dist_dim+self.level_dim, self.feat_dim),
                nn.ReLU(),
                nn.Linear(self.feat_dim, self.n_offsets),
                nn.Sigmoid()
            )




        self.mlp_cov = nn.Sequential(
                nn.Linear(self.feat_dim+self.view_dim+self.cov_dist_dim+self.level_dim, self.feat_dim),
                nn.ReLU(),
                nn.Linear(self.feat_dim, 7*self.n_offsets),
            )
        
        self.mlp_color = nn.Sequential(
                nn.Linear(self.feat_dim+self.view_dim+self.color_dist_dim+self.level_dim+self.appearance_dim, self.feat_dim),
                nn.ReLU(),
                nn.Linear(self.feat_dim, 3*self.n_offsets),
                nn.Sigmoid()
            )

        self.mlp_albedo = nn.Sequential(
                nn.Linear(self.feat_dim+self.view_dim+self.color_dist_dim+self.level_dim+self.appearance_dim, self.feat_dim),
                nn.ReLU(),
                nn.Linear(self.feat_dim, 3*self.n_offsets),
                nn.Sigmoid()
            )


        self.mlp_normal1= nn.Sequential(
                nn.Linear(self.feat_dim+self.view_dim+self.color_dist_dim+self.level_dim+self.appearance_dim, self.feat_dim),
                nn.ReLU(),
                nn.Linear(self.feat_dim, 3*self.n_offsets),
                nn.Sigmoid()
            )

        self.mlp_normal2= nn.Sequential(
                nn.Linear(self.feat_dim+self.view_dim+self.color_dist_dim+self.level_dim+self.appearance_dim, self.feat_dim),
                nn.ReLU(),
                nn.Linear(self.feat_dim, 3*self.n_offsets),
                nn.Sigmoid()
            )

    def eval(self):
        self.mlp_opacity.eval()
        self.mlp_cov.eval()
        self.mlp_color.eval()
        if self.is_pbr:
            self.mlp_albedo.eval()
            self.mlp_matallic.eval()
            self.mlp_roughness.eval()
        if self.normal_detal:
            self.mlp_normal1.eval()
            self.mlp_normal2.eval()


        if self.use_feat_bank:
            self.mlp_feature_bank.eval()
        if self.appearance_dim > 0:
            self.embedding_appearance.eval()


    def train(self):
        self.mlp_opacity.train()
        self.mlp_cov.train()
        self.mlp_color.train()
        if self.is_pbr:
            self.mlp_albedo.train()
            self.mlp_roughness.train()
            self.mlp_matallic.train()
        if self.normal_detal:
            self.mlp_normal1.train()
            self.mlp_normal2.train()

        if self.use_feat_bank:                   
            self.mlp_feature_bank.train()
        if self.appearance_dim > 0:
            self.embedding_appearance.train()


    def capture(self):
        capture_list = [
            self._anchor,
            self._level,
            self._offset,
            self._anchor_feat,
            self.opacity_accum,
            self._scaling,
            self._rotation,
            self._opacity,
            self.offset_gradient_accum,
            self.offset_denom,
            self.anchor_demon,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.mlp_opacity.state_dict(),
            self.mlp_cov.state_dict(),
            self.mlp_color.state_dict(),
            vars(self._saved_training_args) if self._saved_training_args is not None else {},
        ]
        if self.use_feat_bank:
            capture_list.extend([self.mlp_feature_bank.state_dict()])
        else:
            capture_list.extend([None])
        if self.appearance_dim > 0:
            capture_list.extend([self.embedding_appearance.state_dict()])
        else:
            capture_list.extend([None])
        if self.is_pbr:
            capture_list.extend([
                self.mlp_albedo.state_dict(),
                self.mlp_matallic.state_dict(),
                self.mlp_roughness.state_dict()
            ])
        if self.normal_detal:
            capture_list.extend([
                self.mlp_normal1.state_dict(),
                self.mlp_normal2.state_dict()
            ])

        return capture_list

    def capture_numpy(self, skip_sync=False, light=None):
        """Capture all state as numpy arrays.

        NOTE: Jittor 1.3.11 — after rasterizer backward, tensors become CUDA-only
        and .numpy()/.data/jt.sync()/cupy all fail. We work around this by reading
        MLP weights BEFORE backward (caller must invoke at the right time).
        For params that fail, we return zeros as placeholder.

        Args:
            skip_sync: If True, skip jt.sync_all() to avoid SFRL crash after
                       densification. Caller guarantees GPU data is valid.
        """
        from utils.jt_safe import memcpy_to_numpy

        def _to_np(obj):
            if obj is None:
                return None
            if isinstance(obj, jt.Var):
                # sync + .numpy() — standard Jittor GPU→CPU read path
                try:
                    jt.sync_all()
                    return obj.numpy()
                except:
                    pass
                path_log("[FALLBACK] capture_numpy: .numpy() failed, using zeros")
                return np.zeros(obj.shape, dtype=np.float32)
            if isinstance(obj, dict):
                return {k: _to_np(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_to_np(v) for v in obj]
            if isinstance(obj, (int, float, str, bool)):
                return obj
            return None

        raw = self.capture()
        result = _to_np(raw)

        # Phase 31: defensively re-read MLP weights directly from parameters
        # (bypasses potential state_dict caching issues in Jittor 1.3.11)
        if not skip_sync:
            jt.sync_all()  # Phase 28: skip after densification to avoid SFRL crash
        try:
            for mlp_idx, mlp_attr in [(13, 'mlp_opacity'), (14, 'mlp_cov'), (15, 'mlp_color')]:
                mlp = getattr(self, mlp_attr, None)
                if mlp is None:
                    continue
                param_names = ['0.weight', '0.bias', '2.weight', '2.bias']
                fresh_dict = {}
                for pi, p in enumerate(mlp.parameters()):
                    if pi < len(param_names):
                        try:
                            val_direct = p.numpy().copy()
                            fresh_dict[param_names[pi]] = val_direct
                            # Debug: compare with _to_np result
                            orig = result[mlp_idx]
                            if isinstance(orig, dict) and param_names[pi] in orig:
                                val_old = orig[param_names[pi]]
                                diff = np.abs(val_direct - val_old).max()
                                if diff > 1e-10:
                                    from utils.general_utils import path_log
                                    path_log(f"[P31] {mlp_attr}/{param_names[pi]}: direct vs _to_np diff={diff:.6e}")
                        except Exception:
                            # fallback: keep whatever _to_np produced
                            orig = result[mlp_idx]
                            if isinstance(orig, dict) and param_names[pi] in orig:
                                fresh_dict[param_names[pi]] = orig[param_names[pi]]
                            else:
                                fresh_dict[param_names[pi]] = np.zeros(tuple(p.shape), dtype=np.float32)
                result[mlp_idx] = fresh_dict
        except Exception as e:
            from utils.general_utils import path_log
            path_log(f"[capture_numpy] MLP re-read failed: {e}")

        # Phase 64: append Hybridlight state for PBR checkpoint persistence
        if light is not None:
            try:
                light_np = {
                    'base': light.base.numpy().astype(np.float32),
                    'lgtSGs': light.lgtSGs.numpy().astype(np.float32),
                    'specular_reflectance': light.specular_reflectance.numpy().astype(np.float32),
                    'roughness': light.roughness.numpy().astype(np.float32),
                }
                result.append(light_np)
            except Exception as e:
                from utils.general_utils import path_log
                path_log(f"[capture_numpy] light capture failed: {e}")
                result.append(None)

        return result

    def restore_numpy(self, np_data):
        """Restore from numpy-captured state. Rebuilds optimizer (JGaussian pattern).

        Phase 64: Returns (model_restored, light_state) tuple. light_state is None
        for non-PBR checkpoints, or a dict with 'base','lgtSGs','specular_reflectance','roughness'.
        """
        light_state = None
        # Phase 64: detect light state appended at end of np_data list
        if isinstance(np_data, list) and len(np_data) >= 20:
            last = np_data[-1]
            # Light may be stored as dict or numpy object scalar wrapping a dict
            if isinstance(last, np.ndarray) and last.shape == () and last.dtype == np.object_:
                inner = last.item()
                if isinstance(inner, dict) and 'base' in inner:
                    light_state = inner
                    np_data.pop()
                    while np_data and np_data[-1] is None:
                        np_data.pop()
            elif isinstance(last, dict) and 'base' in last:
                light_state = np_data.pop()
                while np_data and np_data[-1] is None:
                    np_data.pop()

        def _to_jt(obj):
            if obj is None:
                return None
            if isinstance(obj, np.ndarray):
                if obj.dtype == np.object_:
                    # Object array from pickle — extract scalar or list
                    if obj.ndim == 0:
                        return _to_jt(obj.item())
                    return [_to_jt(v) for v in obj]
                return jt.array(obj)
            if isinstance(obj, dict):
                return {k: _to_jt(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_to_jt(v) for v in obj]
            # Handle numpy scalar types
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (int, float, str, bool)):
                return obj
            return None

        model_args = _to_jt(np_data)
        # restore() extracts training_args_dict from model_args[-1],
        # reconstructs Namespace, calls training_setup(training_args)
        # which creates a fresh optimizer with correct param references.
        self.restore(model_args, None)
        print("[checkpoint] Restored from numpy checkpoint (optimizer rebuilt)")
        return light_state


    def restore(self, model_args, training_args= None):
        # Backward compat: old checkpoints have 17 items (no use_feat_bank/appearance None pads).
        # Pad to 19 items if needed so we can unpack uniformly.
        if len(model_args) < 19:
            model_args = list(model_args) + [None] * (19 - len(model_args))
        (self._anchor,
        self._level,
        self._offset,
        self._anchor_feat,
        self.opacity_accum,
        self._scaling,
        self._rotation,
        self._opacity,
        self.offset_gradient_accum,
        self.offset_denom,
        self.anchor_demon,
        opt_dict,              # index 11 — optimizer state dict
        self.spatial_lr_scale,
        mlp_opacity,
        mlp_cov,
        mlp_color,
        training_args_dict,    # index 16
        mlp_feature,           # index 17
        mlp_appearance) = model_args[:19]  # index 18

        print("Load model_args Size:",len(model_args))

        # Reconstruct training_args from saved dict (JGaussian pattern)
        if training_args is None and isinstance(training_args_dict, dict) and training_args_dict:
            from argparse import Namespace
            training_args = Namespace(**training_args_dict)

        # Always rebuild optimizer from scratch (momentum reset).
        # Avoids state_dict incompatibility after densification.
        # reset_stats=False preserves densification accumulators from checkpoint.
        if training_args is not None:
            # Enable gradient tracking on restored tensors (jt.array(numpy) defaults to stop_grad)
            self._anchor.requires_grad = True
            self._offset.requires_grad = True
            self._anchor_feat.requires_grad = True
            self._scaling.requires_grad = True
            self._rotation.requires_grad = True
            self._opacity.requires_grad = False
            self.training_setup(training_args, reset_stats=False)
            # Restore optimizer momentum/variance (JGaussian pattern)
            if opt_dict is not None and isinstance(opt_dict, dict) and 'defaults' in opt_dict:
                try:
                    self.optimizer.load_state_dict(opt_dict)
                except Exception as e:
                    print(f"[checkpoint] Optimizer state restore failed: {e}")


        self.mlp_opacity.load_state_dict(mlp_opacity)
        self.mlp_cov.load_state_dict(mlp_cov)
        self.mlp_color.load_state_dict(mlp_color)
        if self.use_feat_bank:
            self.mlp_feature_bank.load_state_dict(mlp_feature)
        if self.appearance_dim > 0:
            self.embedding_appearance.load_state_dict(mlp_appearance)

        


        if self.is_pbr:
            print("Load model_args PBR rendering Param!")
            print("normal_detal",self.normal_detal,len(model_args))
            if not self.normal_detal and len(model_args)>19:
                (mlp_albedo,
                mlp_matallic,
                mlp_roughness
                ) = model_args[19:22]
                self.mlp_albedo.load_state_dict(mlp_albedo)
                self.mlp_roughness.load_state_dict(mlp_roughness)
                self.mlp_matallic.load_state_dict(mlp_matallic)
            elif self.normal_detal and len(model_args)>22:
                (
                mlp_albedo,
                mlp_matallic,
                mlp_roughness,
                mlp_normal1,
                mlp_normal2
                ) = model_args[19:24]
                self.mlp_albedo.load_state_dict(mlp_albedo)
                self.mlp_roughness.load_state_dict(mlp_roughness)
                self.mlp_matallic.load_state_dict(mlp_matallic)
                self.mlp_normal1.load_state_dict(mlp_normal1)
                self.mlp_normal2.load_state_dict(mlp_normal2)

        else:
            if self.normal_detal and len(model_args)>19:
                (mlp_normal1,
                mlp_normal2) = model_args[19:21]
                self.mlp_normal1.load_state_dict(mlp_normal1)
                self.mlp_normal2.load_state_dict(mlp_normal2)
        # Optimizer state dict intentionally NOT loaded — new optimizer
        # created by training_setup with correct param references.
        # Old opt_dict would be incompatible after densification (param count change).

        # CRITICAL: Sync numpy shadows after restore.
        # set_anchor_mask() relies on _anchor_np/_level_np/_extra_level_np
        # to compute per-view LOD filtering. Without this sync, shadows are
        # never initialized after checkpoint load → AttributeError or all-black render.
        self._sync_np_shadows()

        # _extra_level is NOT saved in checkpoint (not a trainable param).
        # Initialize with zeros = no extra level adjustment to LOD computation.
        N = self._anchor.shape[0]
        if not hasattr(self, '_extra_level_np') or self._extra_level_np is None \
           or self._extra_level_np.shape[0] != N:
            self._extra_level_np = np.zeros(N, dtype=np.float32)
        if not hasattr(self, '_extra_level') or self._extra_level is None \
           or self._extra_level.shape[0] != N:
            self._extra_level = jt.array(self._extra_level_np, dtype=jt.float32)

        # _anchor_mask defaults to all-visible (GPU bool tensor, Phase 78)
        if not hasattr(self, '_anchor_mask') or self._anchor_mask is None \
           or self._anchor_mask.shape[0] != N:
            self._anchor_mask = jt.ones(N, dtype='bool')

        # standard_dist, voxel_size, levels are NOT saved in checkpoint
        # (computed during create_from_pcd / set_level / load_ply_sparse_gaussian).
        # They are required by set_anchor_mask() for LOD computation.
        if not hasattr(self, 'standard_dist') or self.standard_dist is None:
            anchor_np = self._anchor_np
            dist_max = np.max(np.sqrt(np.sum((anchor_np[:min(10000, N)] - np.mean(anchor_np, axis=0))**2, axis=1)))
            self.standard_dist = dist_max * 1.5  # extend factor
        if not hasattr(self, 'voxel_size') or self.voxel_size is None:
            bl = getattr(self, 'base_layer', 10)
            self.voxel_size = self.standard_dist / (self.fork ** max(bl if bl > 0 else 10, 5))
        if not hasattr(self, 'levels') or self.levels is None:
            level_np = self._level_np
            self.levels = int(level_np.max() - level_np.min() + 1)
            self.init_level = int(level_np.min())


    @property
    def get_appearance(self):
        return self.embedding_appearance

    @property
    def get_scaling(self):
        return 1.0*self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_anchor(self):
        return self._anchor
    
    @property
    def get_level(self):
        return self._level
    
    @property
    def get_extra_level(self):
        return self._extra_level
        
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_anchor_feat(self):
        return self._anchor_feat
    
    @property
    def get_opacity_mlp(self):
        return self.mlp_opacity   

    @property
    def get_roughness_mlp(self):
        return self.mlp_roughness


    @property
    def get_cov_mlp(self):
        return self.mlp_cov
    
    @property
    def get_color_mlp(self):
        return self.mlp_color

    @property
    def get_albedo_mlp(self):
        return self.mlp_albedo
    
    @property
    def get_normal1_mlp(self):
        return self.mlp_normal1
    
    @property
    def get_normal2_mlp(self):
        return self.mlp_normal2

    @property
    def get_matallic_mlp(self):
        return self.mlp_matallic

    @property
    def get_featurebank_mlp(self):
        return self.mlp_feature_bank
    
    def set_appearance(self, num_cameras):
        if self.appearance_dim > 0:
            self.embedding_appearance = Embedding(num_cameras, self.appearance_dim)

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)    

    def set_coarse_interval(self, coarse_iter, coarse_factor):
        self.coarse_intervals = []
        num_level = self.levels - 1 - self.init_level
        if num_level > 0:
            q = 1/coarse_factor
            a1 = coarse_iter*(1-q)/(1-q**num_level)
            temp_interval = 0
            for i in range(num_level):
                interval = a1 * q ** i + temp_interval
                temp_interval = interval
                self.coarse_intervals.append(interval)

    def set_level(self, points, cameras, scales, dist_ratio=0.95, init_level=-1, levels=-1):
        """Compute LOD levels using pure Jittor GPU ops (matching PyTorch torch.quantile)."""
        # Ensure points is a Jittor tensor on GPU
        if not isinstance(points, jt.Var):
            points = jt.array(points)
        pts = points.float32()

        all_dist_list = []  # accumulate float scalars
        cam_infos_list = []  # accumulate numpy for final cam_infos tensor

        total_cams = sum(len(cameras[s]) for s in scales)
        cam_idx = 0
        print(f"[set_level] Processing {total_cams} cameras across {len(scales)} scales...", flush=True)
        for scale in scales:
            for cam in cameras[scale]:
                cam_idx += 1
                if cam_idx == 1 or cam_idx % 20 == 0:
                    print(f"[set_level] camera {cam_idx}/{total_cams}...", flush=True)
                cam_center = cam.camera_center
                if isinstance(cam_center, jt.Var):
                    cc = cam_center
                else:
                    cc = jt.array(cam_center)
                if cam_idx == 1:
                    print(f"  [DEBUG] cam0 center: {cc.numpy()}, pts range: [{float(pts.min()):.6f}, {float(pts.max()):.6f}]", flush=True)

                cam_infos_list.append([float(cc[0].numpy()), float(cc[1].numpy()),
                                        float(cc[2].numpy()), float(scale)])

                # GPU distance computation (Jittor — matches PyTorch torch.sqrt(torch.sum(...)))
                dist = jt.norm(pts - cc.float32(), dim=1) + jt.float32(1e-10)
                dist_max = jt_quantile(dist, dist_ratio)
                dist_min = jt_quantile(dist, 1.0 - dist_ratio)
                all_dist_list.append(dist_min * scale)
                all_dist_list.append(dist_max * scale)

        self.cam_infos = jt.array(np.array(cam_infos_list, dtype=np.float32))

        # Final quantile on all accumulated distances
        all_dist = jt.array(np.array(all_dist_list, dtype=np.float32))
        dist_max = jt_quantile(all_dist, dist_ratio)
        dist_min = jt_quantile(all_dist, 1.0 - dist_ratio)
        self.standard_dist = dist_max
        if levels == -1:
            self.levels = int(round(math.log2(dist_max / max(dist_min, 1e-10)) / math.log2(self.fork))) + 1
        else:
            self.levels = levels
        if init_level == -1:
            self.init_level = int(self.levels/2)
        else:
            self.init_level = init_level
            
    def octree_sample(self, data, init_pos):
        """Build multi-resolution octree. Uses numpy for grid dedup (SFRL-safe on 8GB).

        NOTE: jt.unique(dim=0) API works correctly (thrust::sort+unique, misc.py:554-780),
        but per-level GPU allocations fragment SFRL on 8GB GPUs, causing cudaErrorIllegalAddress
        in subsequent training. On ≥24GB GPUs, replace with:
            jt.unique(jt.round((data - init_pos) / cur_size), dim=0) * cur_size + init_pos
        """
        t0 = time.time()
        try:
            data_np = data.numpy() if isinstance(data, jt.Var) else data
        except:
            from utils.jt_safe import memcpy_to_numpy
            data_np = memcpy_to_numpy(data) if isinstance(data, jt.Var) else data
        try:
            init_np = init_pos.numpy() if isinstance(init_pos, jt.Var) else init_pos
        except:
            from utils.jt_safe import memcpy_to_numpy
            init_np = memcpy_to_numpy(init_pos) if isinstance(init_pos, jt.Var) else init_pos
        if isinstance(init_np, (float, np.floating)):
            init_np = np.array([init_np])
        positions_list = []
        levels_list = []
        for cur_level in range(self.levels):
            cur_size = self.voxel_size / (float(self.fork) ** cur_level)
            new_np = np.unique(np.round((data_np - init_np) / cur_size), axis=0) * cur_size + init_np
            positions_list.append(new_np)
            levels_list.append(np.full(new_np.shape[0], cur_level, dtype=np.int32))
        self.positions_np = np.concatenate(positions_list, axis=0)
        self.levels_np = np.concatenate(levels_list, axis=0)
        self.positions = jt.array(self.positions_np).float()
        self._level = jt.array(self.levels_np).int32()
        t1 = time.time()
        time_diff = t1 - t0
        print(f"Building octree time: {int(time_diff // 60)} min {time_diff % 60} sec")

    def create_from_pcd(self, points, spatial_lr_scale, logger=None):
        from gaussian_renderer.simple_knn_jt import distCUDA2  # lazy to avoid circular import from gaussian_renderer/__init__.py
        self.spatial_lr_scale = spatial_lr_scale
        # points is numpy array (jt.array doesn't fully copy large arrays to GPU)
        pts_np = points if isinstance(points, np.ndarray) else points.numpy()
        self.centroid = jt.array(np.mean(pts_np, axis=0))  # numpy for correctness
        box_min = np.min(pts_np) * self.extend
        box_max = np.max(pts_np) * self.extend
        box_d = box_max - box_min
        if self.base_layer < 0:
            default_voxel_size = 0.02
            self.base_layer = int(round(math.log2(box_d/default_voxel_size))) - (self.levels//2) + 1
        self.voxel_size = box_d/(float(self.fork) ** self.base_layer)
        self.init_pos = jt.array(box_min).float()
        self.octree_sample(points, self.init_pos)
        jt.sync()  # materialize octree positions before weed_out

        if self.visible_threshold < 0:
            self.visible_threshold = 0.0
            self.positions, self._level, self.visible_threshold, _ = self.weed_out(self.positions, self._level)
        self.positions, self._level, _, _ = self.weed_out(self.positions, self._level)
        jt.sync()  # materialize positions before CUDA kernels (distCUDA2)

        print(f'Branches of Tree: {self.fork}')
        print(f'Base Layer of Tree: {self.base_layer}')
        print(f'Visible Threshold: {self.visible_threshold}')
        print(f'Appearance Embedding Dimension: {self.appearance_dim}') 
        print(f'LOD Levels: {self.levels}')
        print(f'Initial Levels: {self.init_level}')
        print(f'Initial Voxel Number: {self.positions.shape[0]}')
        print(f'Min Voxel Size: {self.voxel_size/(2.0 ** (self.levels - 1))}')
        print(f'Max Voxel Size: {self.voxel_size}')
        if logger is not None:
            logger.info(f'Branches of Tree: {self.fork}')
            logger.info(f'Base Layer of Tree: {self.base_layer}')
            logger.info(f'Visible Threshold: {self.visible_threshold}')
            logger.info(f'Appearance Embedding Dimension: {self.appearance_dim}')
            logger.info(f'LOD Levels: {self.levels}')
            logger.info(f'Initial Levels: {self.init_level}')
            logger.info(f'Initial Voxel Number: {self.positions.shape[0]}')
            logger.info(f'Min Voxel Size: {self.voxel_size/(2.0 ** (self.levels - 1))}')
            logger.info(f'Max Voxel Size: {self.voxel_size}')

        offsets = jt.zeros((self.positions.shape[0] * self.n_offsets, 3)).float()
        anchors_feat = jt.zeros((self.positions.shape[0], self.feat_dim)).float()  # match PT: zero init
        # Compute KNN distances in numpy FIRST (avoids CUDA-only chain: distCUDA2 is jt.code, no CPU version)
        try:
            from scipy.spatial import KDTree
            tree = KDTree(self._anchor_np)
            dists, _ = tree.query(self._anchor_np, k=4)
            # KDTree returns actual Euclidean distances.
            # PT's distCUDA2 returns MEAN OF SQUARED distances.
            # To match: square → mean → sqrt (RMS of 3 nearest neighbors).
            sq_dists = dists[:, 1:] ** 2          # [N, 3] squared distances
            nn_dists = sq_dists.mean(axis=1)       # mean squared distance (matches distCUDA2)
            nn_dists = np.maximum(nn_dists, 1e-7)
            # Phase 43: clamp max KNN distance to 3× voxel_size to prevent giant Gaussians.
            # With sparse anchors, isolated points get huge KNN distances → enormous scales
            # → blocky mosaic artifacts and coverage gaps.
            max_nn_dist = float(self.voxel_size) * 3.0
            nn_dists = np.minimum(nn_dists, max_nn_dist ** 2)  # clamp squared distance
            _scales_init_np = np.log(np.sqrt(nn_dists))  # log(RMS) — matches distCUDA2
            self._scaling_np_temp = np.tile(_scales_init_np[:, None], (1, 6))
            scales = jt.array(self._scaling_np_temp)
        except Exception:
            dist2 = jt.maximum(distCUDA2(self.positions).float(), 0.0000001)
            scales = jt.log(jt.sqrt(dist2))[...,None].repeat(1, 6)
        rots = jt.zeros((self.positions.shape[0], 4))
        rots[:, 0] = 1
        opacities = inverse_sigmoid(0.1 * jt.ones((self.positions.shape[0], 1), dtype=jt.float))

        self.positions.requires_grad = True
        self._anchor = self.positions
        offsets.requires_grad = True
        self._offset = offsets
        anchors_feat.requires_grad = True
        self._anchor_feat = anchors_feat
        scales.requires_grad = True
        self._scaling = scales
        rots.requires_grad = True
        self._rotation = rots
        opacities.requires_grad = False
        self._opacity = opacities
        # self._level remains [N] shape (unsqueeze removed — breaks CUDA-compatibility chain)
        self._extra_level_np = np.zeros(self._anchor.shape[0], dtype=np.float32)
        self._extra_level = jt.array(self._extra_level_np)
        self._anchor_mask = jt.ones(self._anchor.shape[0], dtype='bool')  # GPU bool tensor（Phase 78: 消除 CPU 规避）
        # Numpy shadows for save_ply (jt tensors become CUDA-only, .numpy() fails)
        self._offset_np = np.zeros((self._anchor.shape[0] * self.n_offsets, 3), dtype=np.float32)
        self._anchor_feat_np = np.zeros((self._anchor.shape[0], self.feat_dim), dtype=np.float32)
        _inv_sig_val = np.float32(np.log(0.1 / 0.9))  # inverse_sigmoid(0.1) constant
        self._opacity_np = np.full((self._anchor.shape[0], 1), _inv_sig_val, dtype=np.float32)
        self._rotation_np = np.zeros((self._anchor.shape[0], 4), dtype=np.float32)
        self._rotation_np[:, 0] = 1.0
        # _scaling_np: reuse from numpy-based init above (or compute if fallback was used)
        if hasattr(self, '_scaling_np_temp'):
            self._scaling_np = self._scaling_np_temp
        else:
            self._scaling_np = np.zeros((self._anchor.shape[0], 6), dtype=np.float32)

        # SFRL cleanup: jt.unique/jt.sort in octree_sample/set_level leave GPU alloc fragments
        jt.sync_all(); jt.gc(); jt.gc()

    def map_to_int_level(self, pred_level, cur_level):
        if self.dist2level=='floor':
            int_level = jt.floor(pred_level).int()
            int_level = jt.clamp(int_level, 0, cur_level)
        elif self.dist2level=='round':
            int_level = jt.round(pred_level).int()
            int_level = jt.clamp(int_level, 0, cur_level)
        elif self.dist2level=='ceil':
            int_level = jt.ceil(pred_level).int()
            int_level = jt.clamp(int_level, 0, cur_level)
        elif self.dist2level=='progressive':
            pred_level = jt.clamp(pred_level+1.0, 0.9999, cur_level + 0.9999)
            int_level = jt.floor(pred_level).int()
            self._prog_ratio = jt.frac(pred_level).unsqueeze(dim=1)
            self.transition_mask = (self._level == int_level)
        else:
            raise ValueError(f"Unknown dist2level: {self.dist2level}")
        
        return int_level

    def weed_out(self, anchor_positions, anchor_levels):
        if len(self.cam_infos) == 0:
            return anchor_positions, anchor_levels, 0.0, jt.ones(anchor_positions.shape[0], dtype=jt.bool)
        # Jittor vectorized: broadcast over all cameras simultaneously (GPU, no Python for loop)
        ap = anchor_positions                          # [N, 3]
        al = anchor_levels.reshape(-1)                  # [N] — handles both [N] and [N,1]
        # cam_infos → jt.Var [num_cams, 4]
        cam_infos_jt = jt.array(self.cam_infos)
        ccs = cam_infos_jt[:, :3]                      # [C, 3]
        scs = cam_infos_jt[:, 3:4]                     # [C, 1]
        # Broadcast: [N, 1, 3] - [1, C, 3] → [N, C, 3]
        dist = jt.norm(ap[:, None, :] - ccs[None, :, :], dim=2)  # [N, C]
        dist = dist * scs.squeeze(1)[None, :] + 1e-10              # [N, C]
        pred_level = jt.log2(self.standard_dist / dist) / math.log2(self.fork)  # [N, C]
        int_level = jt.floor(pred_level).int32().clamp(0, self.levels - 1)      # [N, C]
        visible_count = (al[:, None] <= int_level).float32().mean(dim=1)         # [N]
        weed_mask = visible_count > self.visible_threshold                       # [N] bool
        mean_visible = float(visible_count.mean().numpy())
        return anchor_positions[weed_mask], anchor_levels[weed_mask], mean_visible, weed_mask

    def set_anchor_mask(self, cam_center, iteration, resolution_scale,is_training = False):
        # Pure Jittor computation on GPU — no numpy shadows needed
        anchor = self.get_anchor                                    # jt.Var [N, 3]
        level = self._level.reshape(-1)                              # jt.Var [N]
        extra = self._extra_level.reshape(-1)                        # jt.Var [N]
        if isinstance(cam_center, jt.Var):
            cc = cam_center
        else:
            cc = jt.array(cam_center)
        dist = jt.norm(anchor - cc, dim=1) * resolution_scale + jt.float32(1e-10)
        pred_level = jt.log2(self.standard_dist / jt.maximum(dist, jt.float32(1e-10)))
        pred_level = pred_level / math.log2(self.fork) + extra
        pred_level = pred_level.clamp(-1e9, 1e9)  # NaN→clamp limit

        is_training = self.get_color_mlp.training
        if self.progressive and is_training:
            coarse_index = np.searchsorted(self.coarse_intervals, iteration) + 1 + self.init_level
        else:
            coarse_index = self.levels

        int_level = jt.floor(pred_level).int32().clamp(0, coarse_index - 1)
        self._anchor_mask = (level <= int_level)  # GPU bool tensor（Phase 78: 消除 CPU 规避，和 PT 一致）


    def set_anchor_mask_perlevel(self, cam_center, resolution_scale, cur_level):
        # Pure Jittor computation on GPU — no numpy shadows
        anchor = self.get_anchor
        level = self._level.reshape(-1)
        extra = self._extra_level.reshape(-1)
        if isinstance(cam_center, jt.Var):
            cc = cam_center
        else:
            cc = jt.array(cam_center)
        dist = jt.norm(anchor - cc, dim=1) * resolution_scale + jt.float32(1e-10)
        pred_level = jt.log2(self.standard_dist / jt.maximum(dist, jt.float32(1e-10)))
        pred_level = pred_level / math.log2(self.fork) + extra
        pred_level = pred_level.clamp(-1e9, 1e9)
        int_level = jt.floor(pred_level).int32().clamp(0, cur_level)
        self._anchor_mask = (level <= int_level)  # GPU bool tensor（和 PT 一致，Phase 78 修复）

    def training_setup(self, training_args, reset_stats=True):
        self._saved_training_args = training_args  # for cold-resume optimizer rebuild
        # Fix: spatial_lr_scale may be None if not properly saved in checkpoint
        if self.spatial_lr_scale is None:
            self.spatial_lr_scale = 1.0
        self.percent_dense = training_args.percent_dense

        if reset_stats:
            self.opacity_accum = jt.zeros((self.get_anchor.shape[0], 1))
            self.offset_gradient_accum = jt.zeros((self.get_anchor.shape[0]*self.n_offsets, 1))
            self.offset_denom = jt.zeros((self.get_anchor.shape[0]*self.n_offsets, 1))
            self.anchor_demon = jt.zeros((self.get_anchor.shape[0], 1))
            self._extra_level_np = np.zeros(self._anchor.shape[0], dtype=np.float32)
            self._extra_level = jt.array(self._extra_level_np)
            self._anchor_mask = jt.ones(self._anchor.shape[0], dtype='bool')  # GPU bool tensor（Phase 78: 消除 CPU 规避）
        
        l = [
            {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
            {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
            {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
            {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
            {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"}
        ]
        if self.appearance_dim > 0:
            l.append({'params': self.embedding_appearance.parameters(), 'lr': training_args.appearance_lr_init, "name": "embedding_appearance"})

        if self.use_feat_bank:
            l.append({'params': self.mlp_feature_bank.parameters(), 'lr': training_args.mlp_featurebank_lr_init, "name": "mlp_featurebank"})

        if self.is_pbr:
            # Phase 62: use getattr for PBR LR params — Phase 1 checkpoints lack them
            l.append({'params': self.mlp_albedo.parameters(), 'lr': getattr(training_args, 'mlp_albedo_lr_init', 0.075),
                      "name": "mlp_albedo"})
            l.append({'params': self.mlp_matallic.parameters(), 'lr': getattr(training_args, 'mlp_matallic_lr_init', 0.002),  # Phase 65: 0.005→0.002
                     "name": "mlp_matallic"})
            l.append({'params': self.mlp_roughness.parameters(), 'lr': getattr(training_args, 'mlp_roughness_lr_init', 0.005),
                      "name": "mlp_roughness"})

        if self.normal_detal:
            l.append({'params': self.mlp_normal1.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_normal1"})
            l.append({'params': self.mlp_normal2.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_normal2"})

        self.optimizer = jt.optim.Adam(l, lr=0.0, eps=1e-8)
        if self.is_pbr:
            # Phase 65: fallback values aligned with original PyTorch GANG
            self.mlp_albedo_scheduler_args = get_expon_lr_func(lr_init=getattr(training_args, 'mlp_albedo_lr_init', 0.075),
                                                       lr_final=getattr(training_args, 'mlp_albedo_lr_final', 0.00005),
                                                       lr_delay_mult=getattr(training_args, 'mlp_albedo_delay_mult', 0.01),
                                                       max_steps=getattr(training_args, 'mlp_albedo_lr_max_steps', 40000))

            self.mlp_matallic_scheduler_args = get_expon_lr_func(lr_init=getattr(training_args, 'mlp_matallic_lr_init', 0.002),
                                                       lr_final=getattr(training_args, 'mlp_matallic_lr_final', 0.00002),
                                                       lr_delay_mult=getattr(training_args, 'mlp_matallic_delay_mult', 0.01),
                                                       max_steps=getattr(training_args, 'mlp_matallic_lr_max_steps', 40000))


            self.mlp_roughness_scheduler_args = get_expon_lr_func(lr_init=getattr(training_args, 'mlp_roughness_lr_init', 0.005),
                                                                lr_final=getattr(training_args, 'mlp_roughness_lr_final', 0.00005),
                                                                lr_delay_mult=getattr(training_args, 'mlp_roughness_delay_mult', 0.01),
                                                                max_steps=getattr(training_args, 'mlp_roughness_lr_max_steps', 40000))

        self.anchor_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.offset_scheduler_args = get_expon_lr_func(lr_init=training_args.offset_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.offset_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.offset_lr_delay_mult,
                                                    max_steps=training_args.offset_lr_max_steps)

        
        self.mlp_opacity_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_opacity_lr_init,
                                                    lr_final=training_args.mlp_opacity_lr_final,
                                                    lr_delay_mult=training_args.mlp_opacity_lr_delay_mult,
                                                    max_steps=training_args.mlp_opacity_lr_max_steps)
        
        self.mlp_cov_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_cov_lr_init,
                                                    lr_final=training_args.mlp_cov_lr_final,
                                                    lr_delay_mult=training_args.mlp_cov_lr_delay_mult,
                                                    max_steps=training_args.mlp_cov_lr_max_steps)
        
        self.mlp_color_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_color_lr_init,
                                                    lr_final=training_args.mlp_color_lr_final,
                                                    lr_delay_mult=training_args.mlp_color_lr_delay_mult,
                                                    max_steps=training_args.mlp_color_lr_max_steps)

        if self.normal_detal:
            self.mlp_normal1_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_color_lr_init,
                                                lr_final=training_args.mlp_color_lr_final,
                                                lr_delay_mult=training_args.mlp_color_lr_delay_mult,
                                                max_steps=training_args.mlp_color_lr_max_steps)
            self.mlp_normal2_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_color_lr_init,
                                                lr_final=training_args.mlp_color_lr_final,
                                                lr_delay_mult=training_args.mlp_color_lr_delay_mult,
                                                max_steps=training_args.mlp_color_lr_max_steps)
            
        if self.use_feat_bank:
            self.mlp_featurebank_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_featurebank_lr_init,
                                                        lr_final=training_args.mlp_featurebank_lr_final,
                                                        lr_delay_mult=training_args.mlp_featurebank_lr_delay_mult,
                                                        max_steps=training_args.mlp_featurebank_lr_max_steps)
        if self.appearance_dim > 0:
            self.appearance_scheduler_args = get_expon_lr_func(lr_init=training_args.appearance_lr_init,
                                                        lr_final=training_args.appearance_lr_final,
                                                        lr_delay_mult=training_args.appearance_lr_delay_mult,
                                                        max_steps=training_args.appearance_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "offset":
                lr = self.offset_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "anchor":
                lr = self.anchor_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_opacity":
                lr = self.mlp_opacity_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_cov":
                lr = self.mlp_cov_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_color":
                lr = self.mlp_color_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.use_feat_bank and param_group["name"] == "mlp_featurebank":
                lr = self.mlp_featurebank_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.appearance_dim > 0 and param_group["name"] == "embedding_appearance":
                lr = self.appearance_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.is_pbr:
                if param_group["name"] == "mlp_albedo":
                    lr = self.mlp_albedo_scheduler_args(iteration)
                    param_group['lr'] = lr
                if param_group["name"] == "mlp_matallic":
                    lr = self.mlp_matallic_scheduler_args(iteration)
                    param_group['lr'] = lr
                if param_group["name"] == "mlp_roughness":
                    lr = self.mlp_roughness_scheduler_args(iteration)
                    param_group['lr'] = lr
            if self.normal_detal:
                if param_group["name"] == "mlp_normal1":
                    lr = self.mlp_normal1_scheduler_args(iteration)
                    param_group['lr'] = lr
                if param_group["name"] == "mlp_normal2":
                    lr = self.mlp_normal2_scheduler_args(iteration)
                    param_group['lr'] = lr


    def construct_list_of_attributes(self):
        l = []
        l.append('x')
        l.append('y')
        l.append('z')
        l.append('level')
        l.append('extra_level')
        l.append('info')
        for i in range(self.n_offsets * 3):
            l.append('f_offset_{}'.format(i))
        for i in range(self._anchor_feat.shape[1]):
            l.append('f_anchor_feat_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        anchor = self._anchor_np
        levels = self._level_np[:, None]  # restore [N,1] shape for PLY concat
        extra_levels = self._extra_level_np[:, None]
        infos = np.zeros((anchor.shape[0], 1), dtype=np.float32)
        infos[0, 0] = self.voxel_size
        infos[1, 0] = self.standard_dist

        from utils.jt_safe import memcpy_to_numpy

        _tier_log = {}

        def _copy_or_shadow(tensor, shadow_name, shape, transform=None, label=""):
            """3-tier fallback: .numpy() → memcpy_to_numpy → numpy shadow → zeros."""
            t1_error = None
            t2_error = None
            # Tier 1: try direct .numpy()
            try:
                t = tensor.detach()
                if transform:
                    t = transform(t)
                val = t.numpy()
                _tier_log[label] = "T1"
                return val
            except Exception as e1:
                t1_error = type(e1).__name__
            # Tier 2: memcpy D2H via jt.code
            try:
                t = tensor.detach()
                if transform:
                    t = transform(t)
                val = memcpy_to_numpy(t)
                _tier_log[label] = "T2"
                return val
            except Exception as e2:
                t2_error = type(e2).__name__
                path_log(f"[FALLBACK] save_ply {label}: T1+T2 failed (T1:{t1_error} T2:{t2_error}), trying T3")
            # Tier 3: numpy shadow (initial values)
            s = getattr(self, shadow_name, None)
            if s is not None and s.shape[0] == anchor.shape[0]:
                _tier_log[label] = f"T3(shadow) T1:{t1_error} T2:{t2_error}"
                return np.ascontiguousarray(s)
            _tier_log[label] = f"T3(zeros) T1:{t1_error} T2:{t2_error}"
            path_log(f"[FALLBACK] save_ply {label}: T3 shadow also unavailable, using zeros")
            return np.zeros(shape, dtype=np.float32)

        anchor_feats = _copy_or_shadow(self._anchor_feat, '_anchor_feat_np',
                                       (anchor.shape[0], self._anchor_feat.shape[1]),
                                       transform=lambda t: t.detach().contiguous(),
                                       label="anchor_feat")
        offsets = _copy_or_shadow(self._offset, '_offset_np',
                                  (anchor.shape[0], 3 * self.n_offsets),
                                  transform=lambda t: t.detach().reshape(anchor.shape[0], -1).contiguous(),
                                  label="offset")
        opacities = _copy_or_shadow(self._opacity, '_opacity_np',
                                    (anchor.shape[0], 1),
                                    transform=lambda t: t.detach().contiguous(),
                                    label="opacity")
        scales = _copy_or_shadow(self._scaling, '_scaling_np',
                                 (anchor.shape[0], 6),
                                 transform=lambda t: t.detach().contiguous(),
                                 label="scaling")
        rots = _copy_or_shadow(self._rotation, '_rotation_np',
                               (anchor.shape[0], 4),
                               transform=lambda t: t.detach().contiguous(),
                               label="rotation")

        print("\n[PLY save] Data source for each attribute:")
        label_map = {
            "anchor_feat": "anchor_feat (MLP features)",
            "offset": "offset (per-gaussian positions)",
            "opacity": "opacity",
            "scaling": "scaling (KNN distances)",
            "rotation": "rotation (quaternions)",
        }
        status_icon = {
            "T1": "[GPU real]",
            "T2": "[GPU copy]",
        }
        for lbl, name in label_map.items():
            tier_raw = _tier_log.get(lbl, "?")
            if tier_raw.startswith("T1"):
                icon, desc = "[GPU real]", "T1(.numpy) - real trained values"
            elif tier_raw.startswith("T2"):
                icon, desc = "[GPU copy]", "T2(memcpy)  - direct GPU copy"
            elif "T3" in str(tier_raw):
                icon, desc = "[SHADOW] ", f"T3(shadow)  - initial values (stale)"
            else:
                icon, desc = "[?]      ", f"unknown: {tier_raw}"
            print(f"  {icon} {name:35s} {desc}")

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(anchor.shape[0], dtype=dtype_full)
        attributes = np.concatenate((anchor, levels, extra_levels, infos, offsets, anchor_feats, opacities, scales, rots), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def plot_levels(self):
        for level in range(self.levels):
            level_mask = (self._level == level)
            print(f'Level {level}: {jt.sum(level_mask).item()}, Ratio: {jt.sum(level_mask).item()/self._level.shape[0]}')

    def load_ply_sparse_gaussian(self, path):
        plydata = PlyData.read(path)

        anchor = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1).astype(np.float32)
        
        levels = np.asarray(plydata.elements[0]["level"])[... ,np.newaxis].astype(int)
        extra_levels = np.asarray(plydata.elements[0]["extra_level"])[... ,np.newaxis].astype(np.float32)
        self.voxel_size = jt.array(plydata.elements[0]["info"][0]).float()
        self.standard_dist = jt.array(plydata.elements[0]["info"][1]).float()
        
        # self.centroid = jt.mean(anchor, axis=0)

        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis].astype(np.float32)

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((anchor.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((anchor.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        
        # anchor_feat
        anchor_feat_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_anchor_feat")]
        anchor_feat_names = sorted(anchor_feat_names, key = lambda x: int(x.split('_')[-1]))
        anchor_feats = np.zeros((anchor.shape[0], len(anchor_feat_names)))
        for idx, attr_name in enumerate(anchor_feat_names):
            anchor_feats[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        offset_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_offset")]
        offset_names = sorted(offset_names, key = lambda x: int(x.split('_')[-1]))
        offsets = np.zeros((anchor.shape[0], len(offset_names)))
        for idx, attr_name in enumerate(offset_names):
            offsets[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        offsets = offsets.reshape(-1, 3)  # [N, 3*K] → [N*K, 3]

        self._anchor_feat = jt.array(anchor_feats, dtype=jt.float)
        self._level = jt.array(levels, dtype=jt.int32)
        self._extra_level_np = extra_levels.squeeze()
        self._extra_level = jt.array(self._extra_level_np, dtype=jt.float)
        self._offset = jt.array(offsets, dtype=jt.float).contiguous()  # [N*K, 3]
        self._anchor = jt.array(anchor, dtype=jt.float)
        self._scaling = jt.array(scales, dtype=jt.float)
        self._opacity = jt.array(opacities, dtype=jt.float)
        self._rotation = jt.array(rots, dtype=jt.float)
        self._anchor_mask = jt.ones(self._anchor.shape[0], dtype='bool')  # GPU bool tensor（Phase 78: 消除 CPU 规避）        # Numpy shadows for save_ply (already available from numpy load)
        self._anchor_np = anchor
        self._level_np = levels.squeeze()
        self._offset_np = offsets.copy()  # [N*K, 3]
        self._anchor_feat_np = anchor_feats
        self._opacity_np = opacities
        self._scaling_np = scales
        self._rotation_np = rots
        self.levels = jt.max(self._level) - jt.min(self._level) + 1

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                with jt.enable_grad():
                    group["params"][0] = tensor.copy()
                # Lightweight optimizer (Namespace): skip Adam state
                if "m" in group and len(group["m"]) > 0:
                    group["m"][0] = jt.zeros_like(tensor)
                if "values" in group and len(group["values"]) > 0:
                    group["values"][0] = jt.zeros_like(tensor)
                optimizable_tensors[group["name"]] = group["params"][0]
        jt.gc()
        return optimizable_tensors


    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if  'mlp' in group['name'] or \
                'conv' in group['name'] or \
                'feat_base' in group['name'] or \
                'embedding' in group['name']:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            # Lightweight optimizer (Namespace): skip Adam state m/v
            if "m" in group and len(group["m"]) > 0:
                group["m"][0] = jt.concat((group["m"][0], jt.zeros_like(extension_tensor)), dim=0)
            if "values" in group and len(group["values"]) > 0:
                group["values"][0] = jt.concat((group["values"][0], jt.zeros_like(extension_tensor)), dim=0)
            old_tensor = group["params"].pop()
            with jt.enable_grad():
                group["params"].append(jt.concat((old_tensor, extension_tensor), dim=0))
                del old_tensor
            optimizable_tensors[group["name"]] = group["params"][0]

        jt.gc()
        return optimizable_tensors


    def _sync_accumulators_to_jt(self):
        """Sync accumulators (already Jittor tensors — no-op placeholder)."""
        # All accumulators are now Jittor tensors managed via jt.scatter(add)
        pass

    def training_statis_np(self, viewspace_grad_np, opacity, update_filter, offset_selection_mask, anchor_visible_mask):
        """Accumulate gradient/opacity stats for densification (pure Jittor GPU).

        viewspace_grad_np: numpy [G,3] but may be None
        opacity: jt.Var [M*K, 1] per-Gaussian opacity values
        offset_selection_mask: numpy bool [M*K] — which offsets contributed
        anchor_visible_mask: numpy int [M] — visible anchor indices
        """
        if viewspace_grad_np is None or len(viewspace_grad_np) == 0:
            return
        K = self.n_offsets
        N = self._anchor.shape[0]

        # Opacity accumulation via jt.scatter(add) (GPU atomicAdd)
        try:
            op_jt = opacity.clamp(min_v=0).reshape(-1, K).sum(dim=1, keepdims=True)  # [M, 1]
            vis_idx = jt.array(anchor_visible_mask.reshape(-1, 1).astype(np.int32))
            self.opacity_accum = jt.scatter(self.opacity_accum, 0, vis_idx, op_jt, reduce='add')
        except:
            pass

        # Anchor demon via jt.scatter(add) (GPU)
        try:
            ones_jt = jt.ones((len(anchor_visible_mask), 1))
            self.anchor_demon = jt.scatter(self.anchor_demon, 0, vis_idx, ones_jt, reduce='add')
        except:
            pass

        # Offset gradient accumulation (GPU via jt.scatter + jt.norm)
        req_size = N * K
        # Ensure accumulators exist with correct size
        if not hasattr(self, 'offset_gradient_accum') or self.offset_gradient_accum.shape[0] != req_size:
            self.offset_gradient_accum = jt.zeros((req_size, 1))
        if not hasattr(self, 'offset_denom') or self.offset_denom.shape[0] != req_size:
            self.offset_denom = jt.zeros((req_size, 1))

        sel_mask_np = np.asarray(offset_selection_mask, dtype=bool)
        if len(sel_mask_np) != viewspace_grad_np.shape[0]:
            min_len = min(len(sel_mask_np), viewspace_grad_np.shape[0])
            sel_mask_np = sel_mask_np[:min_len]
            viewspace_grad_np = viewspace_grad_np[:min_len]

        if not sel_mask_np.any():
            return

        # Compute grad norm on GPU via Jittor
        grad_jt = jt.array(viewspace_grad_np[sel_mask_np][:, :2].astype(np.float32))  # [S, 2]
        grad_norm_jt = jt.norm(grad_jt, dim=1, keepdims=True)  # [S, 1]

        # Build offset indices via Jittor (matches numpy-only original)
        offset_base = anchor_visible_mask[..., None] * K + jt.arange(K)[None, :]  # [M, K]
        sel_positions = offset_base.reshape(-1)[jt.array(sel_mask_np.astype(np.int32))]  # [S]
        sel_idx = sel_positions.reshape(-1, 1).astype(jt.int32)  # [S, 1]

        # GPU atomicAdd via jt.scatter
        self.offset_gradient_accum = jt.scatter(self.offset_gradient_accum, 0, sel_idx, grad_norm_jt, reduce='add')
        ones_jt2 = jt.ones_like(grad_norm_jt)
        self.offset_denom = jt.scatter(self.offset_denom, 0, sel_idx, ones_jt2, reduce='add')

    # statis grad information to guide liftting.
    def training_statis(self, viewspace_grad, opacity, update_filter, offset_selection_mask, anchor_visible_mask):
        """Accumulate gradient/opacity stats for densification.

        Stats persist across chunks via checkpoint save/restore.
        Numpy shadows initialized from restored jt tensors (not zeros).
        """
        if viewspace_grad is None:
            return
        K = self.n_offsets
        N = self._anchor.shape[0]

        # --- Expand anchor indices [M] → offset indices [M*K] ---
        offset_base = np.repeat(anchor_visible_mask * K, K) + np.tile(np.arange(K), len(anchor_visible_mask))
        sel_positions = offset_base[offset_selection_mask]

        # --- Opacity accumulation (numpy, restored from checkpoint if available) ---
        try:
            op_np = opacity.numpy().reshape(-1, K)
        except:
            op_np = None
        if op_np is not None:
            op_np[op_np < 0] = 0
            try:
                self._opacity_accum_np = getattr(self, '_opacity_accum_np', None)
                if self._opacity_accum_np is None or self._opacity_accum_np.shape[0] != N:
                    try:
                        self._opacity_accum_np = self.opacity_accum.numpy().copy()
                    except:
                        self._opacity_accum_np = np.zeros((N, 1), dtype=np.float32)
                self._opacity_accum_np[anchor_visible_mask] += op_np.sum(axis=1, keepdims=True)
                # Phase 28: defer jt.array() to _sync_accumulators_to_jt() — avoid per-iter SFRL alloc
            except:
                pass

        # --- Anchor demon (numpy, restored from checkpoint if available) ---
        try:
            self._anchor_demon_np = getattr(self, '_anchor_demon_np', None)
            if self._anchor_demon_np is None or self._anchor_demon_np.shape[0] != N:
                try:
                    self._anchor_demon_np = self.anchor_demon.numpy().copy()
                except:
                    self._anchor_demon_np = np.zeros((N, 1), dtype=np.float32)
            self._anchor_demon_np[anchor_visible_mask] += 1
            # Phase 28: defer jt.array()
        except:
            pass

        # --- Offset gradient accumulation (GPU via jt.scatter, Phase 54) ---
        # Reference: JGaussian gaussian_model.py:792 — accum[mask] += grad (jt.Var persistent)
        if len(sel_positions) == 0:
            return
        # Compute gradient norm on GPU (viewspace_grad is jt.Var)
        try:
            grad_norm_np = jt.norm(viewspace_grad[offset_selection_mask, :2], dim=-1).numpy().reshape(-1)
        except:
            grad_norm_np = np.zeros(len(sel_positions), dtype=np.float32)
        sel_flat = sel_positions.ravel().astype(np.int32)
        gn_flat = grad_norm_np.ravel()

        # Phase 57: sel_flat and gn_flat may differ in length if offset_selection_mask
        # and viewspace_grad have mismatched shapes. Truncate to min length.
        if len(sel_flat) != len(gn_flat):
            min_len = min(len(sel_flat), len(gn_flat))
            sel_flat = sel_flat[:min_len]
            gn_flat = gn_flat[:min_len]

        # Phase 57: jt.scatter requires src.shape == x.shape (setitem_op.cc:82).
        # Solution: pad src to full N*K rows using numpy (fast, ~1ms for 1.5M elements),
        # then jt.array → GPU add. Pre-allocated GPU buffers avoid SFRL fragmentation.
        req_size = N * K
        # Allocate/reuse GPU temp buffers
        if not hasattr(self, '_tmp_grad') or self._tmp_grad.shape[0] != req_size:
            self._tmp_grad = jt.zeros((req_size, 1))
        if not hasattr(self, '_tmp_ones') or self._tmp_ones.shape[0] != req_size:
            self._tmp_ones = jt.zeros((req_size, 1))
        if not hasattr(self, 'offset_gradient_accum') or self.offset_gradient_accum.shape[0] != req_size:
            self.offset_gradient_accum = jt.zeros((req_size, 1))
        if not hasattr(self, 'offset_denom') or self.offset_denom.shape[0] != req_size:
            self.offset_denom = jt.zeros((req_size, 1))
        # Build full-size arrays via CPU indexing + GPU copy
        grad_np = np.zeros((req_size, 1), dtype=np.float32)
        grad_np[sel_flat, 0] = gn_flat
        ones_np = np.zeros((req_size, 1), dtype=np.float32)
        ones_np[sel_flat, 0] = 1.0
        self._tmp_grad.update(jt.array(grad_np))
        self._tmp_ones.update(jt.array(ones_np))
        self.offset_gradient_accum += self._tmp_grad
        self.offset_denom += self._tmp_ones
        
    def _prune_anchor_optimizer(self, mask):
        # Convert jt boolean mask to numpy int indices to avoid jt.where CUDA-only crash
        keep_idx = _jt_bool_to_np_indices(mask)
        K = self.n_offsets
        # For offset-level tensors, expand anchor indices to per-offset indices
        # offset has shape [N*K, ...], anchor has shape [N, ...]
        if keep_idx is not None and len(keep_idx) > 0:
            keep_idx_offset = np.repeat(keep_idx * K, K) + np.tile(np.arange(K), len(keep_idx))
        else:
            keep_idx_offset = None
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if  'mlp' in group['name'] or \
                'conv' in group['name'] or \
                'feat_base' in group['name'] or \
                'embedding' in group['name']:
                continue

            idx = keep_idx_offset if group.get('name') == 'offset' else keep_idx

            if idx is not None and len(idx) > 0:
                old_tensor = group["params"][0]
                new_tensor = old_tensor[idx]
                group["params"] = [new_tensor]  # replace in-place for jt.optim.Adam
                del old_tensor
            else:
                old_tensor = group["params"][0]
                new_tensor = old_tensor[:0]
                group["params"] = [new_tensor]
                del old_tensor
            if group['name'] == "scaling":
                scales = group["params"][0]
                temp = scales[:, 3:]
                temp[temp > 0.05] = 0.05
                group["params"][0][:, 3:] = temp
            optimizable_tensors[group["name"]] = group["params"][0]

        jt.gc()
        return optimizable_tensors

    def _rebuild_jittor_from_numpy(self, op_dict):
        """Phase 54: Batch delete-then-create to prevent SFRL reuse segfault.

        Old code replaced params one-at-a-time (del→gc→jt.array), causing SFRL
        to reuse a just-freed block within the same rebuild cycle. On 8GB GPUs
        this triggers segfault when Jittor internals hold stale references.

        Fix: read all old data → delete ALL old → one gc → create ALL new.
        SFRL cannot reuse a block because ALL old blocks are freed together
        and no jt.array() interleaves.
        """
        import numpy as np, gc
        from argparse import Namespace

        def _safe_read(var, fallback=None):
            if var is None: return fallback
            try:
                jt.sync_all()
                return var.numpy()
            except:
                return fallback

        N = self._anchor.shape[0]; K = self.n_offsets

        # Phase 1: Read all old data to numpy, then null all references
        attrs = ['_anchor', '_offset', '_anchor_feat', '_scaling', '_rotation',
                 '_opacity', '_level', '_extra_level']
        saved_np = {}
        for attr in attrs:
            var = getattr(self, attr, None)
            if attr in ('_level', '_extra_level'):
                saved_np[attr] = _safe_read(var, np.zeros(N, dtype=np.int32))
            else:
                saved_np[attr] = _safe_read(var)
            setattr(self, attr, None)
            del var

        # Flush all pending lazy ops + release all old GPU blocks at once
        jt.sync_all()
        gc.collect()
        jt.gc()

        # Phase 2: Create all new jt.Var from saved numpy (no interleaved alloc/free)
        for attr in attrs:
            np_data = saved_np[attr]
            dtype = np.int32 if attr in ('_level', '_extra_level') else np.float32
            v = jt.array(np_data.astype(dtype))
            if attr not in ('_level', '_extra_level'):
                v.requires_grad = True
            setattr(self, attr, v)
        del saved_np

        self.opacity_accum = jt.zeros((N, 1))
        self.offset_gradient_accum = jt.zeros((N * K, 1))
        self.offset_denom = jt.zeros((N * K, 1))
        self.anchor_demon = jt.zeros((N, 1))

        l = [
            {'params': [self._anchor], 'lr': op_dict.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
            {'params': [self._offset], 'lr': op_dict.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
            {'params': [self._anchor_feat], 'lr': op_dict.feature_lr, "name": "anchor_feat"},
            {'params': [self._opacity], 'lr': op_dict.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': op_dict.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': op_dict.rotation_lr, "name": "rotation"},
            {'params': self.mlp_opacity.parameters(), 'lr': op_dict.mlp_opacity_lr_init, "name": "mlp_opacity"},
            {'params': self.mlp_cov.parameters(), 'lr': op_dict.mlp_cov_lr_init, "name": "mlp_cov"},
            {'params': self.mlp_color.parameters(), 'lr': op_dict.mlp_color_lr_init, "name": "mlp_color"},
        ]
        self.optimizer = jt.optim.Adam(l, lr=0.0, eps=1e-8, betas=(0.9, 0.999))
        self.optimizer.n_step = 0
        self._sync_np_shadows()
        jt.sync()
        print(f"  [REBUILD] {N} anchors, params rebuilt one-at-a-time", flush=True)

    def prune_anchor(self,mask):
        valid_points_mask = jt.logical_not(mask)

        # Direct Jittor boolean indexing (GPU, no numpy round-trip)
        self._level = self._level[valid_points_mask]
        self._extra_level = self._extra_level[valid_points_mask]

        optimizable_tensors = self._prune_anchor_optimizer(valid_points_mask)

        self._anchor = optimizable_tensors["anchor"]
        self._offset = optimizable_tensors["offset"]
        self._anchor_feat = optimizable_tensors["anchor_feat"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self._sync_np_shadows()
    def get_remove_duplicates(self, grid_coords, selected_grid_coords_unique, use_chunk = True):
        if use_chunk:
            chunk_size = 4096
            max_iters = grid_coords.shape[0] // chunk_size + (1 if grid_coords.shape[0] % chunk_size != 0 else 0)
            remove_duplicates_list = []
            for i in range(max_iters):
                cur_remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords[i*chunk_size:(i+1)*chunk_size, :]).all(-1).any(-1).view(-1)
                remove_duplicates_list.append(cur_remove_duplicates)
            remove_duplicates = reduce(jt.logical_or, remove_duplicates_list)
        else:
            remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords).all(-1).any(-1).view(-1)
        return remove_duplicates
    
    def anchor_growing(self, iteration, grads, threshold, update_ratio, extra_ratio, extra_up, offset_mask):
        init_length = self.get_anchor.shape[0]
        # Direct Jittor boolean setitem (GPU, contrib.py → ternary)
        grads[jt.logical_not(offset_mask)] = 0.0
        anchor_grads = jt.sum(grads.reshape(-1, self.n_offsets), dim=-1) / (jt.sum(offset_mask.reshape(-1, self.n_offsets), dim=-1) + 1e-6)
        for cur_level in range(self.levels):
            update_value = self.fork ** update_ratio
            level_mask = (self.get_level == cur_level)
            level_ds_mask = (self.get_level == cur_level + 1)
            if level_ds_mask.ndim > 1:
                level_ds_mask = level_ds_mask.squeeze(dim=1)
            if level_mask.ndim > 1:
                level_mask = level_mask.squeeze(dim=1)
            try:
                _level_cnt = int(jt.sum(level_mask).numpy())
            except:
                _level_cnt = 1
            if _level_cnt == 0:
                continue
            cur_size = self.voxel_size / (float(self.fork) ** cur_level)
            ds_size = cur_size / self.fork
            cur_threshold = threshold * (update_value ** cur_level)
            ds_threshold = cur_threshold * update_value
            extra_threshold = cur_threshold * extra_ratio
            candidate_mask = (grads >= cur_threshold) & (grads < ds_threshold)
            candidate_ds_mask = (grads >= ds_threshold)
            candidate_extra_mask = (anchor_grads >= extra_threshold)

            length_inc = self.get_anchor.shape[0] - init_length
            if length_inc > 0 :
                candidate_mask = jt.concat([candidate_mask, jt.zeros(length_inc * self.n_offsets, dtype=jt.bool)], dim=0)
                candidate_ds_mask = jt.concat([candidate_ds_mask, jt.zeros(length_inc * self.n_offsets, dtype=jt.bool)], dim=0)
                candidate_extra_mask = jt.concat([candidate_extra_mask, jt.zeros(length_inc, dtype=jt.bool)], dim=0)

            repeated_mask = level_mask.unsqueeze(1).repeat(1, self.n_offsets).reshape(-1)
            candidate_mask = jt.logical_and(candidate_mask, repeated_mask)
            candidate_ds_mask = jt.logical_and(candidate_ds_mask, repeated_mask)
            if ~self.progressive or iteration > self.coarse_intervals[-1]:
                self._extra_level += extra_up * candidate_extra_mask.float()

            # --- Pure Jittor anchor growing core (GPU, no numpy round-trip) ---
            anchor = self.get_anchor                                    # [N, 3]
            offset = self._offset.reshape(-1, self.n_offsets, 3)       # [N, K, 3]
            scaling = self.get_scaling[:, :3].reshape(-1, 1, 3)         # [N, 1, 3]
            feat = self._anchor_feat                                    # [N, feat_dim]

            # Grid coordinates via Jittor GPU ops
            grid_coords = jt.round((anchor[level_mask] - self.init_pos) / cur_size).int32()  # [M, 3]
            all_xyz = (anchor[:, None, :] + offset * scaling).reshape(-1, 3)  # [N*K, 3]

            # Candidate selection via Jittor boolean indexing (GPU)
            selected_xyz = all_xyz[candidate_mask]  # [G, 3]
            selected_grid_coords = jt.round((selected_xyz - self.init_pos) / cur_size).int32()
            selected_grid_coords_unique, inverse_indices = jt.unique(selected_grid_coords, return_inverse=True, dim=0)
            if selected_grid_coords_unique.shape[0] > 0 and grid_coords.shape[0] > 0:
                remove_duplicates = self.get_remove_duplicates(grid_coords, selected_grid_coords_unique)
                remove_duplicates = jt.logical_not(remove_duplicates)
                candidate_anchor = selected_grid_coords_unique[remove_duplicates] * cur_size + self.init_pos
                new_level = jt.ones(candidate_anchor.shape[0], dtype=jt.int32) * cur_level
                candidate_anchor, new_level, _, weed_mask = self.weed_out(candidate_anchor, new_level)
                remove_duplicates_clone = remove_duplicates.clone()
                remove_duplicates[remove_duplicates_clone] = weed_mask
            else:
                candidate_anchor = jt.zeros([0, 3], dtype=jt.float)
                remove_duplicates = jt.ones([0], dtype=jt.bool)
                new_level = jt.zeros([0], dtype=jt.int32)

            if (~self.progressive or iteration > self.coarse_intervals[-1]) and cur_level < self.levels - 1:
                grid_coords_ds = jt.round((anchor[level_ds_mask] - self.init_pos) / ds_size).int32()
                selected_xyz_ds = all_xyz[candidate_ds_mask]
                selected_grid_coords_ds = jt.round((selected_xyz_ds - self.init_pos) / ds_size).int32()
                selected_grid_coords_unique_ds, inverse_indices_ds = jt.unique(selected_grid_coords_ds, return_inverse=True, dim=0)
                if selected_grid_coords_unique_ds.shape[0] > 0 and grid_coords_ds.shape[0] > 0:
                    remove_duplicates_ds = self.get_remove_duplicates(grid_coords_ds, selected_grid_coords_unique_ds)
                    remove_duplicates_ds = jt.logical_not(remove_duplicates_ds)
                    candidate_anchor_ds = selected_grid_coords_unique_ds[remove_duplicates_ds] * ds_size + self.init_pos
                    new_level_ds = jt.ones(candidate_anchor_ds.shape[0], dtype=jt.int32) * (cur_level + 1)
                    candidate_anchor_ds, new_level_ds, _, weed_ds_mask = self.weed_out(candidate_anchor_ds, new_level_ds)
                    remove_duplicates_ds_clone = remove_duplicates_ds.clone()
                    remove_duplicates_ds[remove_duplicates_ds_clone] = weed_ds_mask
                else:
                    candidate_anchor_ds = jt.zeros([0, 3], dtype=jt.float)
                    remove_duplicates_ds = jt.ones([0], dtype=jt.bool)
                    new_level_ds = jt.zeros([0], dtype=jt.int32)
            else:
                candidate_anchor_ds = jt.zeros([0, 3], dtype=jt.float)
                remove_duplicates_ds = jt.ones([0], dtype=jt.bool)
                new_level_ds = jt.zeros([0], dtype=jt.int32)

            if candidate_anchor.shape[0] + candidate_anchor_ds.shape[0] > 0:

                new_anchor = jt.concat([candidate_anchor, candidate_anchor_ds], dim=0)
                new_level = jt.concat([new_level, new_level_ds]).float()

                # Feature scatter via Jittor GPU
                # feat_per_offset: [N, feat_dim] → [N, K, feat_dim] → [N*K, feat_dim]
                feat_per_offset = feat.unsqueeze(1).repeat(1, self.n_offsets, 1).reshape(-1, self.feat_dim)
                new_feat_candidates = feat_per_offset[candidate_mask]  # [G, feat_dim]
                # scatter-max via reindex_reduce (Jittor 1.3.11 supports "max" reduction)
                dim_size = int(selected_grid_coords_unique.shape[0])
                accumulated = new_feat_candidates.reindex_reduce(
                    "max", [dim_size, self.feat_dim],
                    ["@e0(i0)", "i1"], extras=[inverse_indices]
                )
                new_feat = accumulated[remove_duplicates]
                new_feat_ds = jt.zeros([candidate_anchor_ds.shape[0], self.feat_dim], dtype=jt.float)
                new_feat = jt.concat([new_feat, new_feat_ds], dim=0)
                
                new_scaling = jt.ones_like(candidate_anchor).repeat([1,2]).float()*cur_size # *0.05
                new_scaling_ds = jt.ones_like(candidate_anchor_ds).repeat([1,2]).float()*ds_size # *0.05
                new_scaling = jt.concat([new_scaling, new_scaling_ds], dim=0)
                new_scaling = jt.log(new_scaling)
                
                new_rotation = jt.zeros([candidate_anchor.shape[0], 4], dtype=jt.float)
                new_rotation_ds = jt.zeros([candidate_anchor_ds.shape[0], 4], dtype=jt.float)
                new_rotation = jt.concat([new_rotation, new_rotation_ds], dim=0)
                new_rotation[:,0] = 1.0

                new_opacities = inverse_sigmoid(0.1 * jt.ones((candidate_anchor.shape[0], 1), dtype=jt.float))
                new_opacities_ds = inverse_sigmoid(0.1 * jt.ones((candidate_anchor_ds.shape[0], 1), dtype=jt.float))
                new_opacities = jt.concat([new_opacities, new_opacities_ds], dim=0)

                new_offsets = jt.zeros(candidate_anchor.shape[0] * self.n_offsets, 3).float()
                new_offsets_ds = jt.zeros(candidate_anchor_ds.shape[0] * self.n_offsets, 3).float()
                new_offsets = jt.concat([new_offsets, new_offsets_ds], dim=0)

                new_extra_level = jt.zeros(candidate_anchor.shape[0], dtype=jt.float)
                new_extra_level_ds = jt.zeros(candidate_anchor_ds.shape[0], dtype=jt.float)
                new_extra_level = jt.concat([new_extra_level, new_extra_level_ds])
                
                d = {
                    "anchor": new_anchor,
                    "scaling": new_scaling,
                    "rotation": new_rotation,
                    "anchor_feat": new_feat,
                    "offset": new_offsets,
                    "opacity": new_opacities,
                }   

                temp_anchor_demon = jt.concat([self.anchor_demon, jt.zeros([new_opacities.shape[0], 1]).float()], dim=0)
                del self.anchor_demon
                self.anchor_demon = temp_anchor_demon

                temp_opacity_accum = jt.concat([self.opacity_accum, jt.zeros([new_opacities.shape[0], 1]).float()], dim=0)
                del self.opacity_accum
                self.opacity_accum = temp_opacity_accum

                
                
                optimizable_tensors = self.cat_tensors_to_optimizer(d)
                self._anchor = optimizable_tensors["anchor"]
                self._scaling = optimizable_tensors["scaling"]
                self._rotation = optimizable_tensors["rotation"]
                self._anchor_feat = optimizable_tensors["anchor_feat"]
                self._offset = optimizable_tensors["offset"]
                self._opacity = optimizable_tensors["opacity"]
                self._level = jt.concat([self._level, new_level], dim=0)
                self._extra_level = jt.concat([self._extra_level, new_extra_level], dim=0)
                self._sync_np_shadows()

    def _sync_np_shadows(self):
        """Try to sync numpy shadows from jt tensors (3 tiers: .numpy → memcpy → skip)."""
        from utils.jt_safe import memcpy_to_numpy

        _sync_logged = set()
        def _sync_one(attr_name, np_attr, transform=None):
            try:
                t = getattr(self, attr_name).detach()
                if transform:
                    t = transform(t)
                val = t.numpy()
            except:
                try:
                    t = getattr(self, attr_name).detach()
                    if transform:
                        t = transform(t)
                    val = memcpy_to_numpy(t)
                    if np_attr not in _sync_logged:
                        path_log(f"[FALLBACK] _sync_np_shadows: {np_attr} .numpy() failed, using memcpy_to_numpy")
                        _sync_logged.add(np_attr)
                except:
                    if np_attr not in _sync_logged:
                        path_log(f"[FALLBACK] _sync_np_shadows: {np_attr} T1+T2 failed, keeping old shadow")
                        _sync_logged.add(np_attr)
                    return
            # Squeeze level/extra_level from [N,1] to [N]
            if np_attr in ('_level_np', '_extra_level_np') and val.ndim > 1:
                val = val.squeeze(-1)
            setattr(self, np_attr, val)

        _sync_one('_anchor', '_anchor_np')
        _sync_one('_level', '_level_np')
        _sync_one('_extra_level', '_extra_level_np')
        _sync_one('_offset', '_offset_np')  # both [N*K, 3], no transform needed
        _sync_one('_anchor_feat', '_anchor_feat_np')
        _sync_one('_opacity', '_opacity_np')
        _sync_one('_scaling', '_scaling_np')
        _sync_one('_rotation', '_rotation_np')

    def muti_plane_pruning(self, num=10,std = 2,planer_numer=16):
        # index = np.random.randint(0,3)
        index = 2 
        depth_min = self.get_anchor[:,index].min()
        depth_max = self.get_anchor[:,index].max()
        depth_vector = (depth_max-depth_min)/planer_numer
        depth_mask = jt.zeros(self.get_anchor.shape[0])
        for i in range(planer_numer):
            depth_mask[self.get_anchor[:, 2] <= (depth_min + (1 + i) * depth_vector)] += 1

        mask_point_temp = jt.logical_or(jt.zeros(self.get_anchor.shape[0]), jt.zeros(self.get_anchor.shape[0]))

        for i in range(planer_numer):
            muti_mask = jt.zeros_like(depth_mask)
            muti_mask[depth_mask == (i + 1)] = 1
            try:
                _muti_cnt = int(jt.sum(muti_mask).numpy())
            except:
                _muti_cnt = 0  # assume too few if can't read
            if _muti_cnt < num * 10:
                continue
            pcd_vector = o3d.geometry.PointCloud()
            pcd_vector.points = o3d.utility.Vector3dVector(self.get_anchor[muti_mask==1].detach().numpy())
            cl, ind = pcd_vector.remove_statistical_outlier(num, std)
            mask_t = jt.zeros(muti_mask[muti_mask==1].shape[0])
            mask_t[ind] = 1
            muti_mask[muti_mask==1] = mask_t
            mask_point_temp[muti_mask==1] = True
        return mask_point_temp
    
    def adjust_anchor(self, iteration, check_interval=100, success_threshold=0.8, grad_threshold=0.0002, update_ratio=0.5, extra_ratio=4.0, extra_up=0.25, min_opacity=0.005):
        # Phase 28: sync numpy accumulators to jt.Var (only once per densification interval)
        self._sync_accumulators_to_jt()
        # # adding anchors
        # Use .reshape(-1, 1) for safe shape handling (some tensors may have extra dims)
        grads = self.offset_gradient_accum.reshape(-1, 1) / (self.offset_denom.reshape(-1, 1) + 1e-8)
        # Safe NaN replacement (avoid boolean-indexed assignment crash)
        nan_mask = grads.isnan()
        grads = jt.where(nan_mask, jt.zeros_like(grads), grads)
        grads_norm = jt.norm(grads, dim=-1)
        offset_mask = (self.offset_denom > check_interval*success_threshold*0.5).squeeze(dim=1)

        self.anchor_growing(iteration, grads_norm, grad_threshold, update_ratio, extra_ratio, extra_up, offset_mask)

        # update offset_denom — direct Jittor boolean indexing (GPU)
        self.offset_denom[offset_mask] = 0  # boolean mask setitem → jt.ternary under contrib.py
        padding_offset_demon = jt.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_denom.shape[0], 1],
                                           dtype=jt.int32)
        self.offset_denom = jt.concat([self.offset_denom, padding_offset_demon], dim=0)

        self.offset_gradient_accum[offset_mask] = 0
        padding_offset_gradient_accum = jt.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_gradient_accum.shape[0], 1],
                                           dtype=jt.int32)
        self.offset_gradient_accum = jt.concat([self.offset_gradient_accum, padding_offset_gradient_accum], dim=0)

        # # prune anchors
        prune_mask = (self.opacity_accum < min_opacity*self.anchor_demon).squeeze(dim=1)
        anchors_mask = (self.anchor_demon > check_interval*success_threshold).squeeze(dim=1) # [N, 1]
        prune_mask = jt.logical_and(prune_mask, anchors_mask) # [N]
        keep_mask = jt.logical_not(prune_mask)

        # Direct Jittor boolean indexing (GPU, no numpy round-trip)
        offset_denom = self.offset_denom.view([-1, self.n_offsets])[keep_mask]
        offset_denom = offset_denom.view([-1, 1])
        del self.offset_denom
        self.offset_denom = offset_denom

        offset_gradient_accum = self.offset_gradient_accum.view([-1, self.n_offsets])[keep_mask]
        offset_gradient_accum = offset_gradient_accum.view([-1, 1])
        del self.offset_gradient_accum
        self.offset_gradient_accum = offset_gradient_accum

        # update opacity accum — direct Jittor boolean setitem (GPU)
        self.opacity_accum[anchors_mask] = jt.zeros_like(self.opacity_accum[anchors_mask]).float()
        self.anchor_demon[anchors_mask] = jt.zeros_like(self.anchor_demon[anchors_mask]).float()

        temp_opacity_accum = self.opacity_accum[keep_mask]
        del self.opacity_accum
        self.opacity_accum = temp_opacity_accum

        temp_anchor_demon = self.anchor_demon[keep_mask]
        del self.anchor_demon
        self.anchor_demon = temp_anchor_demon

        if prune_mask.shape[0]>0:
            self.prune_anchor(prune_mask)  # prune 为 True，保留为 False

    def save_mlp_checkpoints(self, path, mode = 'unite'):#split or unite
        mkdir_p(os.path.dirname(path))
        if mode == 'split':
            self.eval()
            opacity_mlp = jt.jit.trace(self.mlp_opacity, (jt.rand(1, self.feat_dim+self.view_dim+self.opacity_dist_dim+self.level_dim)))
            opacity_mlp.save(os.path.join(path, 'opacity_mlp.pt'))

            cov_mlp = jt.jit.trace(self.mlp_cov, (jt.rand(1, self.feat_dim+self.view_dim+self.cov_dist_dim+self.level_dim)))
            cov_mlp.save(os.path.join(path, 'cov_mlp.pt'))
            color_mlp = jt.jit.trace(self.mlp_color, (jt.rand(1, self.feat_dim+self.view_dim+self.color_dist_dim+self.appearance_dim+self.level_dim)))
            color_mlp.save(os.path.join(path, 'color_mlp.pt'))

            if self.normal_detal:
                normal1_mlp = jt.jit.trace(self.mlp_normal1, (jt.rand(1, self.feat_dim + self.view_dim + self.color_dist_dim + self.appearance_dim + self.level_dim)))
                normal1_mlp.save(os.path.join(path, 'normal1_mlp.pt'))

                normal2_mlp = jt.jit.trace(self.mlp_normal2, (jt.rand(1, self.feat_dim + self.view_dim + self.color_dist_dim + self.appearance_dim + self.level_dim)))
                normal2_mlp.save(os.path.join(path, 'normal2_mlp.pt'))

            if self.use_feat_bank:
                feature_bank_mlp = jt.jit.trace(self.mlp_feature_bank, (jt.rand(1, 3+self.level_dim)))
                feature_bank_mlp.save(os.path.join(path, 'feature_bank_mlp.pt'))
            if self.appearance_dim > 0:
                emd = jt.jit.trace(self.embedding_appearance, (jt.zeros((1,), dtype=jt.long)))
                emd.save(os.path.join(path, 'embedding_appearance.pt'))
            if self.is_pbr:
                albedo_mlp = jt.jit.trace(self.mlp_albedo, (jt.rand(1, self.feat_dim + self.view_dim + self.color_dist_dim + self.appearance_dim + self.level_dim)))
                albedo_mlp.save(os.path.join(path, 'albedo_mlp.pt'))
                roughness_mlp = jt.jit.trace(self.mlp_roughness, (jt.rand(1, self.feat_dim+self.view_dim+self.opacity_dist_dim+self.level_dim)))
                roughness_mlp.save(os.path.join(path, 'roughness_mlp.pt'))
                mate_mlp = jt.jit.trace(self.mlp_matallic, (
                    jt.rand(1, self.feat_dim + self.view_dim + self.opacity_dist_dim + self.level_dim)))
                mate_mlp.save(os.path.join(path, 'matallic_mlp.pt'))

            self.train()
        elif mode == 'unite':
            param_dict = {}
            param_dict['opacity_mlp'] = self.mlp_opacity.state_dict()
            param_dict['cov_mlp'] = self.mlp_cov.state_dict()
            param_dict['color_mlp'] = self.mlp_color.state_dict()
            # param_dict['sdf_mlp'] = self.SDF.state_dict()
            
            if self.normal_detal:
                param_dict['normal1_mlp'] = self.mlp_normal1.state_dict()
                param_dict['normal2_mlp'] = self.mlp_normal2.state_dict()

            if self.use_feat_bank:
                param_dict['feature_bank_mlp'] = self.mlp_feature_bank.state_dict()
            if self.appearance_dim > 0:
                param_dict['appearance'] = self.embedding_appearance.state_dict()

            if self.is_pbr:
                param_dict['albedo_mlp'] = self.mlp_albedo.state_dict()
                param_dict['roughness_mlp'] = self.mlp_roughness.state_dict()
                param_dict['matallic_mlp']= self.mlp_matallic.state_dict()


            try:
                jt.save(param_dict, os.path.join(path, 'checkpoints.pkl'))
            except Exception as _e:
                print(f"[WARN] jt.save failed ({_e}), skipping checkpoint save")
        else:
            raise NotImplementedError


    def load_mlp_checkpoints(self, path, mode = 'unite'):#split or unite
        if mode == 'split':
            self.mlp_opacity = jt.jit.load(os.path.join(path, 'opacity_mlp.pt'))
            self.mlp_cov = jt.jit.load(os.path.join(path, 'cov_mlp.pt'))
            self.mlp_color = jt.jit.load(os.path.join(path, 'color_mlp.pt'))
            # self.SDF = jt.jit.load(os.path.join(path, 'sdf_mlp.pt'))
            
            if self.normal_detal:
                self.mlp_normal1 = jt.jit.load(os.path.join(path, 'normal1_mlp.pt'))
                self.mlp_normal2 = jt.jit.load(os.path.join(path, 'normal2_mlp.pt'))

            if self.use_feat_bank:
                self.mlp_feature_bank = jt.jit.load(os.path.join(path, 'feature_bank_mlp.pt'))
            if self.appearance_dim > 0:
                self.embedding_appearance = jt.jit.load(os.path.join(path, 'embedding_appearance.pt'))
            if self.is_pbr:
                self.mlp_albedo = jt.jit.load(os.path.join(path, 'albedo_mlp.pt'))
                self.mlp_roughness = jt.jit.load(os.path.join(path,'roughness_mlp.pt'))
                self.mlp_matallic = jt.jit.load(os.path.join(path,"matallic_mlp.pt"))


        elif mode == 'unite':
            checkpoint = jt.load(os.path.join(path, 'checkpoints.pkl'))
            self.mlp_opacity.load_state_dict(checkpoint['opacity_mlp'])
            self.mlp_cov.load_state_dict(checkpoint['cov_mlp'])
            self.mlp_color.load_state_dict(checkpoint['color_mlp'])
            # self.SDF.load_state_dict(checkpoint['sdf_mlp'])
            if self.normal_detal:
                self.mlp_normal1.load_state_dict(checkpoint['normal1_mlp'])
                self.mlp_normal2.load_state_dict(checkpoint['normal2_mlp'])
            if self.use_feat_bank:
                self.mlp_feature_bank.load_state_dict(checkpoint['feature_bank_mlp'])
            if self.appearance_dim > 0:
                self.embedding_appearance.load_state_dict(checkpoint['appearance'])
            if self.is_pbr:
                self.mlp_albedo.load_state_dict(checkpoint['albedo_mlp'])
                self.mlp_roughness.load_state_dict(checkpoint['roughness_mlp'])
                self.mlp_matallic.load_state_dict(checkpoint['matallic_mlp'])

        else:
            raise NotImplementedError


    def computeNorm(self,scaling, rota,dir_pp_normalized, delta_normal1=None,delta_normal2=None):
        normal_axis = get_minimum_axis(scaling, rota)
        normal_axis, positive = flip_align_view(normal_axis, dir_pp_normalized)
        if self.normal_detal and delta_normal1 is not None:
            delta_normal1 = (delta_normal1-0.5)*2
            delta_normal2 = (delta_normal2-0.5)*2
            delta_normal = jt.stack([delta_normal1, delta_normal2], dim=-1) # (N, 3, 2)
            idx = positive.long()[:,None,:].repeat(1, 3, 1)  # False→0, True→1 (avoids jt.where CUDA-only issue)
            delta_normal = jt.gather(delta_normal, index=idx, dim=-1).squeeze(-1) # (N, 3)
            normal = delta_normal + normal_axis 
            normal = normal/normal.norm(dim=1, keepdim=True) # (N, 3)
            return normal, delta_normal
        else:
            return normal_axis


    
    def position_normal(self,points):
        if self.centroid is None:
            self.centroid = jt.mean(self._anchor, dim=0)
        points_centered = points - self.centroid
        scale = jt.max(jt.norm(points_centered, dim=1))
        normalized_points = (points_centered / scale + 1) / 2 
        return normalized_points

    