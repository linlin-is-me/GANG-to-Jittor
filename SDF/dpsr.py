
import jittor as jt
from jittor import nn
from SDF.utils import spec_gaussian_filter, fftfreqs, img, grid_interp, point_rasterize
import numpy as np

class DPSR(nn.Module):
    def __init__(self, res, sig=10, scale=True, shift=True):
        """
        :param res: tuple of output field resolution. eg., (128,128)
        :param sig: degree of gaussian smoothing
        """
        super(DPSR, self).__init__()
        self.res = res
        self.sig = sig
        self.dim = len(res)
        self.denom = np.prod(res)
        G = spec_gaussian_filter(res=res, sig=sig).float()
        # self.G.requires_grad = False # True, if we also make sig a learnable parameter
        self.omega = fftfreqs(res, dtype=jt.float32)
        self.scale = scale
        self.shift = shift
        # Jittor: register_buffer not available; plain attr auto-registers as parameter
        self.G = G
        
    def execute(self, V, N):
        """
        :param V: (batch, nv, 2 or 3) tensor for point cloud coordinates
        :param N: (batch, nv, 2 or 3) tensor for point normals
        :return phi: (batch, res, res, ...) tensor of output indicator function field
        """
        assert(V.shape == N.shape) # [b, nv, ndims]
        ras_p = point_rasterize(V, N, self.res)  # [b, n_dim, dim0, dim1, dim2]

        # FFT via numpy (DPSR only used for SDF init, not in gradient path)
        ras_p_np = ras_p.numpy()
        ras_s_np = np.fft.rfftn(ras_p_np, axes=(2,3,4))
        ras_s_np = np.transpose(ras_s_np, tuple([0]+list(range(2, self.dim+1))+[self.dim+1, 1]))
        N_np = ras_s_np[..., None] * self.G.numpy()  # [b, dim0, dim1, dim2/2+1, n_dim, 1]

        omega_np = self.omega.numpy()[..., None] * (2 * np.pi)  # [dim0, dim1, dim2/2+1, n_dim, 1]
        omega_sq = omega_np.squeeze(-1)  # [dim0, dim1, dim2/2+1, n_dim]

        DivN_np = np.sum((-1j * N_np[..., 0]) * omega_sq, axis=-1)  # [b, dim0, dim1, dim2/2+1]
        Lap_np = -np.sum(omega_sq ** 2, axis=-1)  # [dim0, dim1, dim2/2+1]
        Phi_np = DivN_np / (Lap_np + 1e-6)  # [b, dim0, dim1, dim2/2+1]

        # Permute and zero DC component
        Phi_np = np.transpose(Phi_np, tuple(list(range(1, self.dim+1)) + [0]))  # [dim0, dim1, dim2/2+1, b]
        Phi_np[tuple([0] * self.dim)] = 0
        Phi_np = np.transpose(Phi_np, tuple([self.dim] + list(range(self.dim))))  # [b, dim0, dim1, dim2/2+1]

        phi_np = np.fft.irfftn(Phi_np, s=self.res, axes=(1,2,3))
        phi = jt.array(phi_np)
        
        if self.shift or self.scale:
            # ensure values at points are zero
            fv = grid_interp(phi.unsqueeze(-1), V, batched=True).squeeze(-1) # [b, nv]
            if self.shift: # offset points to have mean of 0
                offset = jt.mean(fv, dim=-1)  # [b,] 
                phi -= offset.view(*tuple([-1] + [1] * self.dim))
                
            phi = phi.permute(*tuple([list(range(1,self.dim+1)) + [0]]))
            fv0 = phi[tuple([0] * self.dim)]  # [b,]
            phi = phi.permute(*tuple([[self.dim] + list(range(self.dim))]))
            
            if self.scale:
                phi = -phi / jt.abs(fv0.view(*tuple([-1]+[1] * self.dim))) *0.5
        return phi