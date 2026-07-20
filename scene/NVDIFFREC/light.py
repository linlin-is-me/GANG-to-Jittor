# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import os
import numpy as np
import jittor as jt
from jittor import nn

# Jittor texture module — replaces nvdiffrast.torch
from scene.NVDIFFREC.texture import texture as _jt_tex_fn

class _dr_compat:
    """Compatibility wrapper: dr.texture(...) → _jt_tex_fn(...)"""
    @staticmethod
    def texture(*args, **kwargs):
        return _jt_tex_fn(*args, **kwargs)

dr = _dr_compat()

from . import util
from scene.NVDIFFREC.renderutils import specular_cubemap, diffuse_cubemap
from utils.general_utils import get_expon_lr_func
import cv2

import jittor.nn as F
import imageio
from utils.light_utils import DistributionGGX,GeometrySmith,fresnelSchlick


TINY_NUMBER = 1e-6


######################################################################################
# Utility functions
######################################################################################

class cubemap_mip(jt.Function):
    def execute(self, cubemap):
        return util.avg_pool_nhwc(cubemap, (2,2))

    def grad(self, dout):
        res = dout.shape[1] * 2
        out = jt.zeros(6, res, res, dout.shape[-1], dtype=jt.float32)
        for s in range(6):
            gy, gx = jt.meshgrid(jt.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res),
                                    jt.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res),
                                    )
                                    # indexing='ij')
            v = util.safe_normalize(util.cube_to_dir(s, gx, gy))
            out[s, ...] = dr.texture(dout[None, ...] * 0.25, v[None, ...].contiguous(), filter_mode='linear', boundary_mode='cube')
        return out

######################################################################################
# Split-sum environment map light source with automatic mipmap generation
######################################################################################

def compute_energy(lgtSGs):
    lgtLambda = jt.abs(lgtSGs[:, 3:4])       # [M, 1]
    lgtMu = jt.abs(lgtSGs[:, 4:])               # [M, 3]
    energy = lgtMu * 2.0 * np.pi / lgtLambda * (1.0 - jt.exp(-2.0 * lgtLambda))
    return energy

def fibonacci_sphere(samples=1):
    '''
    https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
    '''
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append([x, y, z])
    points = np.array(points)
    return points

class Hybridlight(nn.Module):
    LIGHT_MIN_RES = 16

    MIN_ROUGHNESS = 0.08
    MAX_ROUGHNESS = 0.5

    def __init__(self, base_res = 256,
                 scale = 0.5,
                 bias = 0.25,
                 num_sg = 16,
                 numBrdfSGs = 1,
                 inital_position = None,
                 upper_hemi = False,
                 is_white_light = False,
                 cache_dir = None):
        super(Hybridlight, self).__init__()
        self.mtx = None
        self._cache_dir = cache_dir or '.'

        # Phase 35: Use numpy RNG for deterministic cubemap (cacheable across runs).
        # jt.rand() is non-deterministic → different cache key each run.
        np.random.seed(42)
        base_np = (np.random.rand(6, base_res, base_res, 3).astype(np.float32) * scale + bias)
        base = jt.array(base_np)
        self.base = base
        # self.register_parameter('env_base', self.base)

        self.numLgtSGs = num_sg

        self.white_light = is_white_light
        # Initialize SG params in numpy (Jittor .data assignment doesn't work)
        if is_white_light:
            print("SG is white light!!")
            lgt_np = np.random.randn(num_sg, 8).astype(np.float32)  # pos(3)+lobe(3)+lambda(1)+mu(1)
            spec_np = np.random.randn(numBrdfSGs, 1).astype(np.float32)
            energy_offset = 6  # lambda at col 6, mu at col 7
        else:
            lgt_np = np.random.randn(num_sg, 10).astype(np.float32)  # pos(3)+lobe(3)+lambda(1)+mu(3)
            lgt_np[:, -2:] = lgt_np[:, -3:-2]  # copy to last 2 cols
            spec_np = np.random.randn(numBrdfSGs, 3).astype(np.float32)
            energy_offset = 6  # lambda at col 6, mu at cols 7-9
        spec_np = np.abs(spec_np)

        self.get_env = None

        if inital_position is not None:
            lgt_np[:, :3] = np.array(inital_position)
        else:
            print("random SG light position inital!!!")

        # make sure lambda is not too close to zero
        lgt_np[:, energy_offset:energy_offset+1] = 20. + np.abs(lgt_np[:, energy_offset:energy_offset+1] * 100.)
        # make sure total energy is around 1
        energy_np = compute_energy(jt.array(lgt_np[:, energy_offset-3:])).numpy()
        lgt_np[:, energy_offset+1:] = np.abs(lgt_np[:, energy_offset+1:]) / np.sum(energy_np, axis=0, keepdims=True) * 2. * np.pi
        energy_check = compute_energy(jt.array(lgt_np[:, energy_offset-3:])).numpy()
        print('init envmap energy: ', np.sum(energy_check))

        lobes = fibonacci_sphere(self.numLgtSGs).astype(np.float32)
        lgt_np[:, 3:6] = lobes

        self.upper_hemi = upper_hemi
        if self.upper_hemi:
            print('Restricting lobes to upper hemisphere!')
            self.restrict_lobes_upper = lambda lgtSGs: jt.concat(
                (lgtSGs[..., :1], jt.abs(lgtSGs[..., 1:2]), lgtSGs[..., 2:]), dim=-1)
            lgt_np[:, 1:2] = np.abs(lgt_np[:, 1:2])

        # Convert to JT at the end (avoids .data assignment issues)
        self.lgtSGs = jt.array(lgt_np)
        self.specular_reflectance = jt.array(spec_np)

        # optimize
        roughness = [np.random.uniform(1.5, 2.0) for i in range(numBrdfSGs)]           # big roughness
        roughness = np.array(roughness).astype(dtype=np.float32).reshape((numBrdfSGs, 1))  # [K, 1]
        print('init SG roughness: ', 1.0 / (1.0 + np.exp(-roughness)))
        self.roughness = jt.array(roughness)


    def training_setup(self,training_args):

        l = [
            {'params': self.base, 'lr': training_args.env_map_init, "name": "Envmap"},
            {'params': self.lgtSGs, 'lr': training_args.sg_init, "name": "SGLight"},
            {'params': self.roughness,'lr': training_args.sg_init, "name": "sg_roughness"},
            {'params': self.specular_reflectance, 'lr': training_args.sg_init, "name": "specular_reflectance"},
        ]

        self.optimizer = jt.optim.Adam(l, lr=0.0, eps=1e-8)

        self.env_light_scheduler = get_expon_lr_func(lr_init=training_args.env_map_init,
                                                         lr_final=training_args.env_map_final,
                                                         lr_delay_mult=training_args.appearance_lr_delay_mult,
                                                         max_steps=training_args.iterations)

        self.sg_light_scheduler = get_expon_lr_func(lr_init=training_args.sg_init,
                                                         lr_final=training_args.sg_final,
                                                         lr_delay_mult=training_args.appearance_lr_delay_mult,
                                                         max_steps=training_args.iterations)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "Envmap":
                lr = self.env_light_scheduler(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "SGLight" or param_group["name"] == "specular_reflectance" or param_group["name"] == "sg_roughness":
                lr = self.sg_light_scheduler(iteration)
                param_group['lr'] = lr


    def xfm(self, mtx):
        self.mtx = mtx

    def clone(self):
        return Hybridlight(self.base.clone().detach())

    def clamp_(self, min=None, max=None):
        self.base.clamp_(min, max)

    def get_mip(self, roughness):
        mask = (roughness < self.MAX_ROUGHNESS).float()
        branch_a = (jt.clamp(roughness, self.MIN_ROUGHNESS, self.MAX_ROUGHNESS) - self.MIN_ROUGHNESS) / (self.MAX_ROUGHNESS - self.MIN_ROUGHNESS) * (len(self.specular) - 2)
        branch_b = (jt.clamp(roughness, self.MAX_ROUGHNESS, 1.0) - self.MAX_ROUGHNESS) / (1.0 - self.MAX_ROUGHNESS) + len(self.specular) - 2
        return mask * branch_a + (1.0 - mask) * branch_b


    def load_from_numpy(self, light_dict):
        """Phase 64: Load light state from numpy dict (from checkpoint npz)."""
        if light_dict is None:
            return
        self.base = jt.array(light_dict['base'].astype(np.float32))
        self.lgtSGs = jt.array(light_dict['lgtSGs'].astype(np.float32))
        self.specular_reflectance = jt.array(light_dict['specular_reflectance'].astype(np.float32))
        self.roughness = jt.array(light_dict['roughness'].astype(np.float32))
        self.numLgtSGs = self.lgtSGs.shape[0]
        print("  Light state restored from checkpoint")

    def load_light(self, filepath,is_training=False):
        assert(filepath.endswith('.npy'))

        print("load Light paramer!!")


        light_dict = np.load(filepath, allow_pickle=True)
        lgtSG = light_dict.item()["lgtSGs"]
        base = light_dict.item()["base"]
        specular_reflectance = light_dict.item()["specular_reflectance"]
        sg_roughness = light_dict.item()["sg_roughness"]

        self.lgtSGs = jt.array(lgtSG)
        self.base = jt.array(base)
        self.specular_reflectance = jt.array(specular_reflectance)
        self.roughness = jt.array(sg_roughness)
        self.numLgtSGs = self.lgtSGs.shape[0]

    def save_light(self,path):
        result = {}
        result["lgtSGs"] = self.lgtSGs.detach().numpy()
        result["base"] = self.base.detach().numpy()
        result["specular_reflectance"] = self.specular_reflectance.detach().numpy()
        result["sg_roughness"] = self.roughness.detach().numpy()
        np.save(path,result)

    def build_mips(self, cutoff=0.99):
        # Phase 35: Check disk cache first to avoid slow numpy specular_cubemap.
        # specular_cubemap at res=256 takes hours in pure numpy.
        # The cubemap depends only on self.base (SG initialization), which is
        # deterministic given the same random seed. Cache to disk for reuse.
        import os, hashlib
        cache_dir = getattr(self, '_cache_dir', '.')
        os.makedirs(os.path.join(cache_dir, 'pbr_cache'), exist_ok=True)
        base_hash = hashlib.md5(self.base.numpy().tobytes()).hexdigest()[:12]
        cache_path = os.path.join(cache_dir, 'pbr_cache', f'mips_{self.base.shape[1]}_{base_hash}.npz')

        if os.path.exists(cache_path):
            data = dict(np.load(cache_path, allow_pickle=True))
            n_mips = int(data['n_mips'])
            self.specular = [jt.array(data[f'specular_{i}']) for i in range(n_mips)]
            self.diffuse = jt.array(data['diffuse'])
            return

        # Compute mip chain (Jittor ops — fast with no_grad)
        with jt.no_grad():
            self.specular = [self.base]
            while self.specular[-1].shape[1] > self.LIGHT_MIN_RES:
                self.specular += [cubemap_mip.apply(self.specular[-1])]

        self.diffuse = diffuse_cubemap(self.specular[-1])

        # Phase 38: specular_cubemap now CUDA-accelerated via jt.code — no performance issue.
        n_levels = len(self.specular)
        for idx in range(n_levels - 1):
            if n_levels > 2:
                roughness = (idx / (n_levels - 2)) * (self.MAX_ROUGHNESS - self.MIN_ROUGHNESS) + self.MIN_ROUGHNESS
            else:
                roughness = self.MIN_ROUGHNESS
            self.specular[idx] = specular_cubemap(self.specular[idx], roughness, cutoff)
        self.specular[-1] = specular_cubemap(self.specular[-1], 1.0, cutoff)

        # Save to cache
        try:
            cache_data = {'n_mips': len(self.specular),
                          'diffuse': self.diffuse.numpy()}
            for i, s in enumerate(self.specular):
                cache_data[f'specular_{i}'] = s.numpy()
            np.savez_compressed(cache_path, **cache_data)
        except Exception:
            pass  # Cache write failure is non-fatal

    def regularizer(self):
        white = (self.base[..., 0:1] + self.base[..., 1:2] + self.base[..., 2:3]) / 3.0
        return jt.mean(jt.abs(self.base - white))

    def compute_env_envmap(self,filename=None,res=[512, 1024],return_img = False):
        # cubemap_to_latlong
        gy, gx = jt.meshgrid(
            jt.linspace(0.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0]),
            jt.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1]),
            indexing="ij",
        )

        sintheta, costheta = jt.sin(gy * np.pi), jt.cos(gy * np.pi)
        sinphi, cosphi = jt.sin(gx * np.pi), jt.cos(gx * np.pi)

        reflvec = jt.stack(
            (sintheta * sinphi, costheta, -sintheta * cosphi), dim=-1
        )  # [H, W, 3]
        color = dr.texture(
            self.base[None, ...],
            reflvec[None, ...].contiguous(),
            filter_mode="linear",
            boundary_mode="cube",
        )[
            0
        ]  # [H, W, 3]
        if return_img:
            return color
        else:
            cv2.imwrite(filename, color.clamp(0.0).numpy()[..., ::-1])

    def compute_SG_envmap(self,SGs= None, filename=None, res=[512, 1024],return_img = False, upper_hemi=False):
        H,W = res
        # exactly same convetion as Mitsuba, check envmap_convention.png
        if upper_hemi:
            phi, theta = jt.meshgrid(
                [jt.linspace(0., np.pi / 2., H), jt.linspace(-0.5 * np.pi, 1.5 * np.pi, W)])
        else:
            phi, theta = jt.meshgrid([jt.linspace(0., np.pi, H), jt.linspace(-0.5 * np.pi, 1.5 * np.pi, W)])

        viewdirs = jt.stack([jt.cos(theta) * jt.sin(phi), jt.cos(phi), jt.sin(theta) * jt.sin(phi)],
                               dim=-1)  # [H, W, 3]

        if SGs is None:
            lgtSGs = self.lgtSGs.clone().detach()
        else:
            lgtSGs = SGs

        viewdirs = viewdirs.unsqueeze(-2)  # [..., 1, 3]
        # [M, 7] ---> [..., M, 7]
        dots_sh = list(viewdirs.shape[:-2])
        M = lgtSGs.shape[0]
        lgtSGs = lgtSGs.view([1, ] * len(dots_sh) + [M, 10]).expand(dots_sh + [M, 10])
        # sanity
        # [..., M, 3]
        lgtSGLobes = lgtSGs[..., 3:6] / (jt.norm(lgtSGs[..., 3:6], dim=-1, keepdim=True))
        lgtSGLambdas = jt.abs(lgtSGs[..., 6:7])
        lgtSGMus = jt.abs(lgtSGs[..., -3:])  # positive values
        # [..., M, 3]
        rgb = lgtSGMus * jt.exp(lgtSGLambdas * (jt.sum(viewdirs * lgtSGLobes, dim=-1, keepdim=True) - 1.))
        rgb = jt.sum(rgb, dim=-2)  # [..., 3]
        envmap = rgb.reshape((H, W, 3))
        if return_img:
            return envmap
        else:
            cv2.imwrite(filename, envmap.clamp(0.0).numpy())

    def lightRender(self, points, normal, albedo, roughness, metallic, viewdirs,is_env = True):
        
        N,_ = normal.shape

        #specular
   
        if self.numLgtSGs > 0:
            specular_rgb_sg, diffuse_rgb_sg = self.sg_render(normal,viewdirs,points,albedo,roughness,metallic)
        else:
            specular_rgb_sg = jt.zeros_like(albedo)
            diffuse_rgb_sg = jt.zeros_like(albedo)
        # Phase 91: force sync+gc to free sg_render intermediates (~8GB [N,16,3] tensors)
        # before envmap cubemap rendering allocates more.
        if self.numLgtSGs > 0:
            jt.sync_all(); jt.gc(); jt.gc()

        if is_env:
            normals = normal.reshape(1, N, 3)
            view_dirs = viewdirs.reshape(1, N, 3)
            albedo = albedo.reshape(1, N, 3)
            roughness = roughness.reshape(1, N, 1)
            
            metallic = metallic.reshape(1, N, 1)
            
            diff_col  = albedo * (1.0 - metallic)

            ref_dirs = (2.0 * (normals * view_dirs).sum(-1, keepdims=True).clamp(0.0) * normals - view_dirs)

            diffuse_light = dr.texture(self.diffuse[None, ...], normals[None, ...].contiguous(), filter_mode='linear',
                                    boundary_mode='cube')
            diffuse_rgb = diffuse_light * diff_col
            jt.sync_all(); jt.gc(); jt.gc()  # Phase 91: free diffuse cubemap intermediates

            # specular
            NoV = jt.clamp(util.dot(view_dirs, normals), 1e-4, 1.0)
            fg_uv = jt.concat((NoV, roughness), dim=-1)  # [1, N, 2]
            if not hasattr(self, '_FG_LUT'):
                self._FG_LUT = jt.array(
                    np.fromfile('scene/NVDIFFREC/irrmaps/bsdf_256_256.bin', dtype=np.float32).reshape(1, 256, 256,2),
                    dtype=jt.float32)
            fg_lookup = dr.texture(
                self._FG_LUT,  # [1, 256, 256, 2]
                fg_uv[None,...].contiguous(),  # [1, N, 2]
                filter_mode="linear",
                boundary_mode="clamp",
            )  # [1, N, 2]
            
            miplevel = self.get_mip(roughness)

            spec = dr.texture(self.specular[0][None, ...], ref_dirs[None,...].contiguous(),
                                mip=list(m[None, ...] for m in self.specular[1:]), mip_level_bias=miplevel.permute(0,2,1),
                                filter_mode='linear-mipmap-linear', boundary_mode='cube')
            jt.sync_all(); jt.gc(); jt.gc()  # Phase 91: free specular mipmap cubemap intermediates

            F0 = (1.0 - metallic) * 0.04 + albedo * metallic
           

            reflectance = F0 * fg_lookup[..., 0:1] + fg_lookup[..., 1:2]  # [1,N, 3]


            specular_rgb = spec * reflectance  # [1, N, 3]

            extras = {"specular_rgb": specular_rgb[0,0],"diffuse_rgb": diffuse_rgb[0,0]}


            render_rgb = diffuse_rgb[0,0] + diffuse_rgb_sg + specular_rgb[0,0]+specular_rgb_sg  # [N, 3]
        
        else:
            extras = {"specular_rgb": specular_rgb_sg,"diffuse_rgb": diffuse_rgb_sg}
            render_rgb = diffuse_rgb_sg + specular_rgb_sg  # [N, 3]

        return render_rgb, extras
    
    def sg_render(self,normal,viewdirs,points,albedo,roughness,metallic):
        N, _ = normal.shape
        M = self.lgtSGs.shape[0]

        roughness = jt.clamp(roughness,0.00001,1)

        normal_sg = normal.unsqueeze(-2).expand([N, M, 3])
        viewdirs_sg = viewdirs.unsqueeze(-2).expand([N, M, 3])
        point_sg = points.unsqueeze(-2).expand([N, M, 3])
        roughness_sg = roughness.unsqueeze(-2).expand([N, M, 1])
        albedo_sg = albedo.unsqueeze(-2).expand([N, M, 3])
 
        metallic_sg = metallic.unsqueeze(-2).expand([N, M, 3])
        lgtSGs = self.lgtSGs.unsqueeze(0).expand([N, M, 10])  # # [N, M, 10]

        #### note: sanity
        lgtSGPosition = lgtSGs[..., :3]  # [N, M, 3]
        lgtSGLobes = lgtSGs[..., 3:6] / (
                jt.norm(lgtSGs[..., 3:6], dim=-1, keepdim=True) + TINY_NUMBER)  # [N, M, 3]
        lgtSGLambdas = jt.abs(lgtSGs[..., 6:7])
        lgtSGMus = jt.abs(lgtSGs[..., -3:])  # positive values

        decay_weight = compute_weight(point_sg, lgtSGPosition)  # [N, M]

        # NDF
        brdfSGLobes = normal_sg  # use normal as the brdf SG lobes
        inv_roughness_pow4 = 1. / (roughness_sg * roughness_sg * roughness_sg * roughness_sg)  # [N, M, 1]

        brdfSGLambdas = (2. * inv_roughness_pow4)  # [N, M, 1]
        brdfSGMus = (inv_roughness_pow4 / np.pi)  # [N, M, 3]
       
        # perform spherical warping
        v_dot_lobe = jt.sum(brdfSGLobes * viewdirs_sg, dim=-1, keepdim=True)  # [N, M, 1]
        ### note: for numeric stability
        v_dot_lobe = v_dot_lobe.clamp(0.0)  # [N, M, 1]
        warpBrdfSGLobes = 2 * v_dot_lobe * brdfSGLobes - viewdirs_sg  # [N, M, 3]
        warpBrdfSGLobes = warpBrdfSGLobes / (
                    jt.norm(warpBrdfSGLobes, dim=-1, keepdim=True) + TINY_NUMBER)  # [N, M, 3]
        # warpBrdfSGLambdas = brdfSGLambdas / (4 * jt.abs(jt.sum(brdfSGLobes * viewdirs, dim=-1, keepdim=True)) + TINY_NUMBER)
       
        warpBrdfSGLambdas = brdfSGLambdas / (4 * v_dot_lobe + TINY_NUMBER)  # # [N, M, 1] can be huge
        warpBrdfSGMus = brdfSGMus  # [N, M, 3]

        # add fresnel and geometric terms; apply the smoothness assumption in SG paper
        new_half = warpBrdfSGLobes + viewdirs_sg  # [N, M, 3]
        new_half = new_half / (jt.norm(new_half, dim=-1, keepdim=True) + TINY_NUMBER)  # [N, M, 3]
        v_dot_h = jt.sum(viewdirs_sg * new_half, dim=-1, keepdim=True)  # [N, M, 1]
        ### note: for numeric stability
        v_dot_h = v_dot_h.clamp(0.0)  # [N, M, 1]
        specular_reflectance = albedo_sg

        F = specular_reflectance + (1. - specular_reflectance) * jt.pow(2.0, -(
                5.55473 * v_dot_h + 6.8316) * v_dot_h)  # [N, M, 1]
        

        dot1 = jt.sum(warpBrdfSGLobes * normal_sg, dim=-1, keepdim=True)  # [N, M, 1]
        ### note: for numeric stability
        dot1 = dot1.clamp(0.0)  # [N, M, 1]
        dot2 = jt.sum(viewdirs_sg * normal_sg, dim=-1, keepdim=True)  # [N, M, 1]
        ### note: for numeric stability
        dot2 = dot2.clamp(0.0)
        k = (roughness_sg + 1.) * (roughness_sg + 1.) / 8.  # [N, M, 1]
        G1 = dot1 / (dot1 * (1 - k) + k + TINY_NUMBER)  # [N, M, 1]
        G2 = dot2 / (dot2 * (1 - k) + k + TINY_NUMBER)  # [N, M, 1]
        G = G1 * G2  # [N, M, 1]

        Moi = F * G / (4 * dot1 * dot2 + TINY_NUMBER)  # [N, M, 1]
        warpBrdfSGMus = warpBrdfSGMus * Moi  # [N, M, 3]

        # multiply with light sg
        final_lobes, final_lambdas, final_mus = lambda_trick(lgtSGLobes, lgtSGLambdas, lgtSGMus,
                                                                warpBrdfSGLobes, warpBrdfSGLambdas, warpBrdfSGMus)
        mu_cos = 32.7080
        lambda_cos = 0.0315
        alpha_cos = 31.7003
        lobe_prime, lambda_prime, mu_prime = lambda_trick(normal_sg, lambda_cos, mu_cos,
                                                            final_lobes, final_lambdas, final_mus)
        # print("lobe_prime",lobe_prime.max(), lambda_prime.max(), mu_prime.max(),)
        dot1 = jt.sum(lobe_prime * normal_sg, dim=-1, keepdim=True)  # [N, M, 1]
        dot2 = jt.sum(final_lobes * normal_sg, dim=-1, keepdim=True)  # [N, M, 1]

        # Phase 91: float64 subtraction to avoid 115× cancellation amplification.
        # term1 ≈ term2 → tiny float32 errors in each term get magnified.
        H1 = hemisphere_int(lambda_prime, dot1)
        H2 = hemisphere_int(final_lambdas, dot2)
        specular_rgb_sg = (mu_prime.float64() * H1.float64() -
                           final_mus.float64() * (alpha_cos * H2.float64())).float32()

        specular_rgb_sg = (specular_rgb_sg * decay_weight.unsqueeze(-1)).sum(dim=-2)  # [N, 3]
        specular_rgb_sg = specular_rgb_sg.clamp(0.0)  # [N, 3]

        # Phase 91: sync+gc to free specular intermediates (~8GB [N,16,3] tensors)
        # before allocating diffuse intermediates.
        jt.sync_all(); jt.gc(); jt.gc()

        # diffuse color
        diffuse = (1-metallic_sg)*albedo_sg / np.pi  # [N, M, 3]
       
        # multiply with light sg
        # .narrow(dim=-2, start=0, length=1) → Jittor slicing
        final_lobes = lgtSGLobes[..., :1, :]  # [N, M, 3]
        final_mus = lgtSGMus[..., :1, :] * diffuse
        final_lambdas = lgtSGLambdas[..., :1, :]

        # now multiply with clamped cosine, and perform hemisphere integral
        lobe_prime, lambda_prime, mu_prime = lambda_trick(normal_sg, lambda_cos, mu_cos,
                                                            final_lobes, final_lambdas, final_mus)

        dot1 = jt.sum(lobe_prime * normal_sg, dim=-1, keepdim=True)
        dot2 = jt.sum(final_lobes * normal_sg, dim=-1, keepdim=True)
        diffuse_rgb_sg = (mu_prime.float64() * hemisphere_int(lambda_prime, dot1).float64() -
                           final_mus.float64() * (alpha_cos * hemisphere_int(final_lambdas, dot2).float64())).float32()
        
        diffuse_rgb_sg = (diffuse_rgb_sg * decay_weight.unsqueeze(-1)).sum(dim=-2)  # [N, 3]
        diffuse_rgb_sg = diffuse_rgb_sg.clamp(0.0, 1.0)
        return specular_rgb_sg, diffuse_rgb_sg



######################################################################################
# Load and store
######################################################################################

# Load from latlong .HDR file

def read_hdr(path: str) -> np.ndarray:
    """Reads an HDR map from disk.  

    Args:
        path (str): Path to the .hdr file.

    Returns:
        numpy.ndarray: Loaded (float) HDR map with RGB channels in order.
    """
    with open(path, "rb") as h:
        buffer_ = np.frombuffer(h.read(), np.uint8)
    bgr = cv2.imdecode(buffer_, cv2.IMREAD_UNCHANGED)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb

def load_env_hdr(fn, sg_pth = None,res=256,numLgtSGs=16, is_sg=True, scale=1.0):
    
    if fn[-4:] == ".hdr":
        with open(fn, "rb") as h:
            buffer_ = np.frombuffer(h.read(), np.uint8)
        bgr = cv2.imdecode(buffer_, cv2.IMREAD_UNCHANGED)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        rgb = imageio.imread(fn)[:,:,:3]

    rgb = np.clip(rgb/255.0,0.0,1.0)*255.0
    latlong_img =jt.array(rgb)


    cubemap = util.latlong_to_cubemap(latlong_img, [res, res])

    l = Hybridlight(res)

    if sg_pth is not None:
        print("Load Pretrain Light Param!")
        l.load_light(sg_pth)
    
    l.base = cubemap
    l.build_mips()

    if is_sg:
        latlong_img_sg = jt.array(normalize_hdr(rgb,method="exposure",alpha=1.0))
        lgtSGs = fit_sg_envmap(numLgtSGs,latlong_img_sg)
        # Jittor: reconstruct lgtSGs tensor (no .data slice assignment)
        l.lgtSGs = jt.concat([l.lgtSGs[:, :3], lgtSGs], dim=1)

    return l




def load_env(fn, sg_path, res,numLgtSGs,is_sg=True, scale=1.0):
    return load_env_hdr(fn, sg_path, res, numLgtSGs,is_sg,scale)


def save_env_map(fn, light):
    assert isinstance(light, Hybridlight), "Can only save EnvironmentLight currently"
    if isinstance(light, Hybridlight):
        color = util.cubemap_to_latlong(light.base, [512, 1024])
    util.save_image_raw(fn, color.detach().numpy())

######################################################################################
# Create trainable env map with random initialization
######################################################################################

def create_trainable_env_rnd(base_res, scale=0.5, bias=0.25):
    base = jt.rand(6, base_res, base_res, 3, dtype=jt.float32) * scale + bias
    return Hybridlight(base)

def extract_env_map(light, resolution=[512, 1024]):
    assert isinstance(light, Hybridlight), "Can only save EnvironmentLight currently"
    color = util.cubemap_to_latlong(light.base, resolution)
    return color




def saturate_dot(a: jt.Var, b: jt.Var) -> jt.Var:
    return (a * b).sum(dim=-1, keepdim=True).clamp(1e-4, 1.0)





def hemisphere_int(lambda_val, cos_beta):
    lambda_val = lambda_val + TINY_NUMBER

    inv_lambda_val = 1. / lambda_val
    t = jt.sqrt(lambda_val) * (1.6988 + 10.8438 * inv_lambda_val) / (
                1. + 6.2201 * inv_lambda_val + 10.2415 * inv_lambda_val * inv_lambda_val)

    ### note: for numeric stability
    inv_a = jt.exp(-t)
    mask = (cos_beta >= 0).float()
    inv_b = jt.exp(-t * cos_beta.clamp(0.0))
    s1 = (1. - inv_a * inv_b) / (1. - inv_a + inv_b - inv_a * inv_b)
    b = jt.exp(t * cos_beta.clamp(-float('inf'), 0.0))
    s2 = (b - inv_a) / ((1. - inv_a) * (b + 1.))
    s = mask * s1 + (1. - mask) * s2

    A_b = 2. * np.pi / lambda_val * (jt.exp(-lambda_val) - jt.exp(-2. * lambda_val))
    A_u = 2. * np.pi / lambda_val * (1. - jt.exp(-lambda_val))

    return A_b * (1. - s) + A_u * s



def compute_weight(point_sg,lgtSGPosition):
    diff = (lgtSGPosition-point_sg)
    squared_diff = diff ** 2  # 每个维度的差值平方
    distance = jt.sqrt(squared_diff.sum(dim=-1)) #(N, K)
    return jt.exp(-0.4*distance)


def lambda_trick(lobe1, lambda1, mu1, lobe2, lambda2, mu2):
    # assume lambda1 << lambda2
    ratio = lambda1 / lambda2

    dot = jt.sum(lobe1 * lobe2, dim=-1, keepdim=True)
    tmp = jt.sqrt(ratio * ratio + 1. + 2. * ratio * dot)
    tmp = jt.minimum(tmp, ratio + 1.)

    lambda3 = lambda2 * tmp
    lambda1_over_lambda3 = ratio / tmp
    lambda2_over_lambda3 = 1. / tmp
    # Phase 91: float64 for the entire cancellation chain.
    # tmp ≈ ratio+1.0 → subtraction loses ~7 bits in f32 → λ2(>20000)× amplification.
    # When both lambdas are tensors (BRDF×light SG path), compute ratio/dot/tmp/diff/exp in float64.
    if hasattr(lambda1, 'float64') and hasattr(lambda2, 'float64'):
        r_f64 = lambda1.float64() / lambda2.float64()
        d_f64 = jt.sum(lobe1.float64() * lobe2.float64(), dim=-1, keepdim=True)
        t_f64 = jt.sqrt(r_f64 * r_f64 + 1.0 + 2.0 * r_f64 * d_f64)
        t_f64 = jt.minimum(t_f64, r_f64 + 1.0)
        diff = lambda2.float64() * (t_f64 - r_f64 - 1.0)

        tmp_f32 = t_f64.float32()
        ratio_f32 = r_f64.float32()
        lambda3 = lambda2 * tmp_f32
        lambda1_over_lambda3 = ratio_f32 / tmp_f32
        lambda2_over_lambda3 = 1.0 / tmp_f32
    else:
        # Scalar lambdas (cosine lobe: λ=0.0315, μ=32.7080).
        # lambda1 or lambda2 is a Python float → ratio is a jt.Var broadcast from scalar.
        # diff = lambda2 * (tmp - ratio - 1.0) — still cancellation-prone if lambda2 is large.
        ratio = lambda1 / lambda2
        dot = jt.sum(lobe1 * lobe2, dim=-1, keepdim=True)
        tmp = jt.sqrt(ratio * ratio + 1. + 2. * ratio * dot)
        tmp = jt.minimum(tmp, ratio + 1.)
        lambda3 = lambda2 * tmp
        lambda1_over_lambda3 = ratio / tmp
        lambda2_over_lambda3 = 1. / tmp
        # Use float64 for diff if lambda2 is a tensor (cosine lobe × final_lambdas)
        if hasattr(lambda2, 'float64'):
            diff = lambda2.float64() * (tmp.float64() - ratio.float64() - 1.0)
        else:
            diff = lambda2 * (tmp - ratio - 1.0)

    final_lobes = lambda1_over_lambda3 * lobe1 + lambda2_over_lambda3 * lobe2
    final_lambdas = lambda3
    final_mus = mu1 * mu2 * jt.exp(diff).float32()

    return final_lobes, final_lambdas, final_mus


def get_envmap_dirs(res = [512, 1024]):
    # Phase 54c: jt.meshgrid in Jittor 1.3.11 doesn't support indexing= kwarg.
    # Use numpy meshgrid then convert to jt.Var (same pattern as SDF/utils.py).
    gy_np, gx_np = np.meshgrid(
        np.linspace(0.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0]),
        np.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1]),
        indexing="ij",
    )
    gy = jt.array(gy_np, dtype=jt.float32)
    gx = jt.array(gx_np, dtype=jt.float32)

    sintheta, costheta = jt.sin(gy * np.pi), jt.cos(gy * np.pi)
    sinphi, cosphi = jt.sin(gx * np.pi), jt.cos(gx * np.pi)

    reflvec = jt.stack((sintheta * sinphi, costheta, -sintheta * cosphi), dim=-1)  # [H, W, 3]
    return reflvec


def fit_sg_envmap(numLgtSGs,hdri):

    # Initialize in numpy (avoids Jittor .data assignment issues)
    lgt_np = np.random.randn(numLgtSGs, 7).astype(np.float32)
    lgt_np[:, 3:4] = 20. + np.abs(lgt_np[:, 3:4] * 100.)
    energy_np = compute_energy(jt.array(lgt_np)).numpy()
    lgt_np[:, 4:] = np.abs(lgt_np[:, 4:]) / np.sum(energy_np, axis=0, keepdims=True) * 2. * np.pi
    lobes = fibonacci_sphere(numLgtSGs).astype(np.float32)
    lgt_np[:, :3] = lobes
    lgtSGs = jt.array(lgt_np)

    optimizer = jt.optim.Adam([lgtSGs,], lr=1e-2)

    H, W = hdri.shape[:2]
    
    N_iter = 2000
    
    for step in range(N_iter):
        env_map = SG2Envmap(lgtSGs, H, W)
        loss = jt.mean((env_map - hdri) * (env_map - hdri))
        optimizer.zero_grad()
        optimizer.backward(loss)
        optimizer.step()

        if step % 200 == 0:
            try:
                loss_val = loss.item()
            except RuntimeError:
                loss_val = float('nan')  # CUDA-only tensor fallback
            print('step: {}, loss: {}'.format(step, loss_val))
    
    return lgtSGs
       


def SG2Envmap(lgtSGs, H=512, W=1024, upper_hemi=False):
    # exactly same convetion as Mitsuba, check envmap_convention.png
    if upper_hemi:
        phi, theta = jt.meshgrid([jt.linspace(0., np.pi/2., H), jt.linspace(-0.5*np.pi, 1.5*np.pi, W)])
    else:
        phi, theta = jt.meshgrid([jt.linspace(0., np.pi, H), jt.linspace(-0.5*np.pi, 1.5*np.pi, W)])

    viewdirs = jt.stack([jt.cos(theta) * jt.sin(phi), jt.cos(phi), jt.sin(theta) * jt.sin(phi)],
                           dim=-1)    # [H, W, 3]
    # print(viewdirs[0, 0, :], viewdirs[0, W//2, :], viewdirs[0, -1, :])
    # print(viewdirs[H//2, 0, :], viewdirs[H//2, W//2, :], viewdirs[H//2, -1, :])
    # print(viewdirs[-1, 0, :], viewdirs[-1, W//2, :], viewdirs[-1, -1, :])

    # lgtSGs = lgtSGs.clone().detach()
    # Jittor: no .to(device) needed, all tensors share same device
    viewdirs = viewdirs.unsqueeze(-2)  # [..., 1, 3]
    # [M, 7] ---> [..., M, 7]
    dots_sh = list(viewdirs.shape[:-2])
    M = lgtSGs.shape[0]
    lgtSGs = lgtSGs.view([1,]*len(dots_sh)+[M, 7]).expand(dots_sh+[M, 7])
    # sanity
    # [..., M, 3]
    lgtSGLobes = lgtSGs[..., :3] / (jt.norm(lgtSGs[..., :3], dim=-1, keepdim=True) + TINY_NUMBER)
    lgtSGLambdas = jt.abs(lgtSGs[..., 3:4])
    lgtSGMus = jt.abs(lgtSGs[..., -3:])  # positive values
    # [..., M, 3]
    rgb = lgtSGMus * jt.exp(lgtSGLambdas * (jt.sum(viewdirs * lgtSGLobes, dim=-1, keepdim=True) - 1.))
    rgb = jt.sum(rgb, dim=-2)  # [..., 3]
    envmap = rgb.reshape((H, W, 3))
    
    return envmap


def normalize_hdr(envmap, method="max", alpha=1.0, exposure=1.0, gamma=2.2):
    """
    Normalize HDR environment map.
    
    Parameters:
    - envmap: HDR environment map as a numpy array (H, W, C).
    - method: Normalization method ("max", "log", "exposure").
    - alpha: Parameter for log normalization (used when method="log").
    - exposure: Exposure parameter (used when method="exposure").
    - gamma: Gamma correction value.

    Returns:
    - norm_envmap: Normalized HDR environment map.
    """
    # Convert to luminance (optional, depends on use case)
    luminance = np.mean(envmap, axis=-1)
    # luminance = envmap
    
    if method == "max":
        # Maximum value normalization
        max_val = np.max(luminance)
        # max_val = np.percentile(luminance,80)
        norm_envmap = envmap / max_val if max_val > 0 else envmap
    elif method == "log":
        # Log-based normalization
        max_val = np.max(luminance)
        # max_val = np.percentile(luminance,80)
        norm_envmap = np.log(1 + alpha * envmap) / np.log(1 + alpha * max_val)
    elif method == "exposure":
        # Exposure adjustment normalization
        norm_envmap = envmap / (2 ** exposure)
    else:
        raise ValueError("Invalid normalization method. Choose 'max', 'log', or 'exposure'.")
    
    # Apply gamma correction
    if gamma > 0:
        norm_envmap = np.clip(norm_envmap, 0, 1) ** (1.0 / gamma)
    
    return norm_envmap
