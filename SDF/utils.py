import jittor as jt
import numpy as np


def spec_gaussian_filter(res, sig):
    omega = fftfreqs(res, dtype=jt.float64) # [dim0, dim1, dim2, d]
    dis = jt.sqrt(jt.sum(omega ** 2, dim=-1))
    filter_ = jt.exp(-0.5*((sig*2*dis/res[0])**2)).unsqueeze(-1).unsqueeze(-1)
    filter_.requires_grad = False

    return filter_


def fftfreqs(res, dtype=jt.float32, exact=True):
    """
    Helper function to return frequency tensors
    :param res: n_dims int tuple of number of frequency modes
    :return:
    """

    n_dims = len(res)
    freqs = []
    for dim in range(n_dims - 1):
        r_ = res[dim]
        freq = np.fft.fftfreq(r_, d=1/r_)
        freqs.append(jt.array(freq, dtype=dtype))
    r_ = res[-1]
    if exact:
        freqs.append(jt.array(np.fft.rfftfreq(r_, d=1/r_), dtype=dtype))
    else:
        freqs.append(jt.array(np.fft.rfftfreq(r_, d=1/r_)[:-1], dtype=dtype))
    omega = jt.meshgrid(freqs)
    omega = list(omega)
    omega = jt.stack(omega, dim=-1)

    return omega


def img(x, deg=1): # imaginary of tensor (assume last dim: real/imag)
    """
    multiply tensor x by i ** deg
    """
    deg %= 4
    if deg == 0:
        res = x
    elif deg == 1:
        res = x[..., [1, 0]]
        res[..., 0] = -res[..., 0]
    elif deg == 2:
        res = -x
    elif deg == 3:
        res = x[..., [1, 0]]
        res[..., 1] = -res[..., 1]
    return res


def grid_interp(grid, pts, batched=True):
    """
    :param grid: tensor of shape (batch, *size, in_features)
    :param pts: tensor of shape (batch, num_points, dim) within range (0, 1)
    :return values at query points
    """
    if not batched:
        grid = grid.unsqueeze(0)
        pts = pts.unsqueeze(0)
    dim = pts.shape[-1]
    bs = grid.shape[0]
    size = jt.array(grid.shape[1:-1], dtype=jt.float32)
    cubesize = 1.0 / size

    ind0 = jt.floor(pts / cubesize).int64()  # (batch, num_points, dim)
    ind1 = (jt.ceil(pts / cubesize) % size).int64()  # periodic wrap-around
    ind01 = jt.stack((ind0, ind1), dim=0)  # (2, batch, num_points, dim)
    # Jittor meshgrid workaround: use numpy to generate binary corner coordinates
    com_np = np.array(np.meshgrid(*[[0, 1]] * dim, indexing='ij')).reshape(dim, -1).T  # [2**dim, dim]
    com_ = jt.array(com_np, dtype=jt.int32)
    dim_ = jt.arange(dim).repeat(com_.shape[0], 1)  # (2**dim, dim)
    ind_ = ind01[com_, ..., dim_]  # (2**dim, dim, batch, num_points)
    ind_n = ind_.permute(2, 3, 0, 1)  # (batch, num_points, 2**dim, dim)
    ind_b = jt.arange(bs).expand(ind_n.shape[1], ind_n.shape[2], bs).permute(2, 0, 1)  # (batch, num_points, 2**dim)
    # latent code on neighbor nodes
    if dim == 2:
        lat = grid.clone()[ind_b, ind_n[..., 0], ind_n[..., 1]]  # (batch, num_points, 2**dim, in_features)
    else:
        lat = grid.clone()[
            ind_b, ind_n[..., 0], ind_n[..., 1], ind_n[..., 2]]  # (batch, num_points, 2**dim, in_features)

    # weights of neighboring nodes
    xyz0 = ind0.float32() * cubesize  # (batch, num_points, dim)
    xyz1 = (ind0.float32() + 1) * cubesize  # (batch, num_points, dim)
    xyz01 = jt.stack((xyz0, xyz1), dim=0)  # (2, batch, num_points, dim)
    pos = xyz01[com_, ..., dim_].permute(2, 3, 0, 1)  # (batch, num_points, 2**dim, dim)
    pos_ = xyz01[1 - com_, ..., dim_].permute(2, 3, 0, 1)  # (batch, num_points, 2**dim, dim)
    pos_ = pos_.float32()
    dxyz_ = jt.abs(pts.unsqueeze(-2) - pos_) / cubesize  # (batch, num_points, 2**dim, dim)
    weights = jt.prod(dxyz_, dim=-1, keepdim=False)  # (batch, num_points, 2**dim)
    query_values = jt.sum(lat * weights.unsqueeze(-1), dim=-2)  # (batch, num_points, in_features)
    if not batched:
        query_values = query_values.squeeze(0)

    return query_values




def scatter_to_grid(inds, vals, size):
    """
    Scatter update values into empty tensor of size size.
    :param inds: (#values, dims)
    :param vals: (#values)
    :param size: tuple for size. len(size)=dims
    """
    dims = inds.shape[1]
    assert(inds.shape[0] == vals.shape[0])
    assert(len(size) == dims)
    result = jt.zeros(*size).view(-1).float32()  # flatten
    # flatten inds
    fac = [np.prod(size[i+1:]) for i in range(len(size)-1)] + [1]
    fac = jt.array(fac).float32()
    inds_fold = jt.sum(inds*fac, dim=-1)  # [#values,]
    result = jt.scatter(result, 0, inds_fold, vals, reduce='add')
    result = result.view(*size)
    return result

def point_rasterize(pts, vals, size):
    """
    :param pts: point coords, tensor of shape (batch, num_points, dim) within range (0, 1)
    :param vals: point values, tensor of shape (batch, num_points, features)
    :param size: len(size)=dim tuple for grid size
    :return rasterized values (batch, features, res0, res1, res2)
    """
    dim = pts.shape[-1]
    assert (pts.shape[:2] == vals.shape[:2])
    assert (pts.shape[2] == dim)
    size_list = list(size)
    size = jt.array(size, dtype=jt.float32)
    cubesize = 1.0 / size
    bs = pts.shape[0]
    nf = vals.shape[-1]
    npts = pts.shape[1]

    ind0 = jt.floor(pts / cubesize).int64()  # (batch, num_points, dim)
    ind1 = (jt.ceil(pts / cubesize) % size).int64()  # periodic wrap-around
    ind01 = jt.stack((ind0, ind1), dim=0)  # (2, batch, num_points, dim)
    # Jittor meshgrid workaround: use numpy to generate binary corner coordinates
    com_np = np.array(np.meshgrid(*[[0, 1]] * dim, indexing='ij')).reshape(dim, -1).T  # [2**dim, dim]
    com_ = jt.array(com_np, dtype=jt.int32)
    dim_ = jt.arange(dim).repeat(com_.shape[0], 1)  # (2**dim, dim)
    ind_ = ind01[com_, ..., dim_]  # (2**dim, dim, batch, num_points)
    ind_n = ind_.permute(2, 3, 0, 1)  # (batch, num_points, 2**dim, dim)
    # ind_b = jt.arange(bs).expand(ind_n.shape[1], ind_n.shape[2], bs).permute(2, 0, 1) # (batch, num_points, 2**dim)
    ind_b = jt.arange(bs).expand(ind_n.shape[1], ind_n.shape[2], bs).permute(2, 0,
                                                                                            1)  # (batch, num_points, 2**dim)

    # weights of neighboring nodes
    xyz0 = ind0.float32() * cubesize  # (batch, num_points, dim)
    xyz1 = (ind0.float32() + 1) * cubesize  # (batch, num_points, dim)
    xyz01 = jt.stack((xyz0, xyz1), dim=0)  # (2, batch, num_points, dim)
    pos = xyz01[com_, ..., dim_].permute(2, 3, 0, 1)  # (batch, num_points, 2**dim, dim)
    pos_ = xyz01[1 - com_, ..., dim_].permute(2, 3, 0, 1)  # (batch, num_points, 2**dim, dim)
    pos_ = pos_.float32()
    dxyz_ = jt.abs(pts.unsqueeze(-2) - pos_) / cubesize  # (batch, num_points, 2**dim, dim)
    weights = jt.prod(dxyz_, dim=-1, keepdim=False)  # (batch, num_points, 2**dim)

    ind_b = ind_b.unsqueeze(-1).unsqueeze(-1)  # (batch, num_points, 2**dim, 1, 1)
    ind_n = ind_n.unsqueeze(-2)  # (batch, num_points, 2**dim, 1, dim)
    ind_f = jt.arange(nf).view(1, 1, 1, nf, 1)  # (1, 1, 1, nf, 1)
    # ind_f = jt.arange(nf).view(1, 1, 1, nf, 1)  # (1, 1, 1, nf, 1)

    ind_b = ind_b.expand(bs, npts, 2 ** dim, nf, 1)
    ind_n = ind_n.expand(bs, npts, 2 ** dim, nf, dim)
    ind_f = ind_f.expand(bs, npts, 2 ** dim, nf, 1)
    inds = jt.concat([ind_b, ind_f, ind_n], dim=-1)  # (batch, num_points, 2**dim, nf, 1+1+dim)

    # weighted values
    vals = weights.unsqueeze(-1) * vals.unsqueeze(-2)  # (batch, num_points, 2**dim, nf)

    inds = inds.view(-1, dim + 2).permute(1, 0).int64()  # (1+dim+1, bs*npts*2**dim*nf)
    vals = vals.reshape(-1)  # (bs*npts*2**dim*nf)
    tensor_size = [bs, nf] + size_list
    raster = scatter_to_grid(inds.permute(1, 0), vals, [bs, nf] + size_list)

    return raster
