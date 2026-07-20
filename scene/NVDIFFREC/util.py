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
import jittor.nn as F
import imageio


#----------------------------------------------------------------------------
# Vector operations
#----------------------------------------------------------------------------

def dot(x: jt.Var, y: jt.Var) -> jt.Var:
    return jt.sum(x*y, -1, keepdim=True)

def reflect(x: jt.Var, n: jt.Var) -> jt.Var:
    return 2*dot(x, n)*n - x

def length(x: jt.Var, eps: float =1e-20) -> jt.Var:
    return jt.sqrt(jt.clamp(dot(x,x), eps, float('inf'))) # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN

def safe_normalize(x: jt.Var, eps: float =1e-20) -> jt.Var:
    return x / length(x, eps)

def to_hvec(x: jt.Var, w: float) -> jt.Var:
    return F.pad(x, pad=(0,1), mode='constant', value=w)

#----------------------------------------------------------------------------
# sRGB color transforms
#----------------------------------------------------------------------------

def _rgb_to_srgb(f: jt.Var) -> jt.Var:
    mask = (f <= 0.0031308).float()
    return mask * (f * 12.92) + (1.0 - mask) * (jt.pow(jt.clamp(f, 0.0031308), 1.0/2.4) * 1.055 - 0.055)

def rgb_to_srgb(f: jt.Var) -> jt.Var:
    assert f.shape[-1] == 3 or f.shape[-1] == 4
    out = jt.concat((_rgb_to_srgb(f[..., 0:3]), f[..., 3:4]), dim=-1) if f.shape[-1] == 4 else _rgb_to_srgb(f)
    assert out.shape[0] == f.shape[0] and out.shape[1] == f.shape[1] and out.shape[2] == f.shape[2]
    return out

def _srgb_to_rgb(f: jt.Var) -> jt.Var:
    mask = (f <= 0.04045).float()
    return mask * (f / 12.92) + (1.0 - mask) * jt.pow((jt.clamp(f, 0.04045) + 0.055) / 1.055, 2.4)

def srgb_to_rgb(f: jt.Var) -> jt.Var:
    assert f.shape[-1] == 3 or f.shape[-1] == 4
    out = jt.concat((_srgb_to_rgb(f[..., 0:3]), f[..., 3:4]), dim=-1) if f.shape[-1] == 4 else _srgb_to_rgb(f)
    assert out.shape[0] == f.shape[0] and out.shape[1] == f.shape[1] and out.shape[2] == f.shape[2]
    return out

def reinhard(f: jt.Var) -> jt.Var:
    return f/(1+f)

#-----------------------------------------------------------------------------------
# Metrics (taken from jaxNerf source code, in order to replicate their measurements)
#
# https://github.com/google-research/google-research/blob/301451a62102b046bbeebff49a760ebeec9707b8/jaxnerf/nerf/utils.py#L266
#
#-----------------------------------------------------------------------------------

def mse_to_psnr(mse):
  """Compute PSNR given an MSE (we assume the maximum pixel value is 1)."""
  return -10. / np.log(10.) * np.log(mse)

def psnr_to_mse(psnr):
  """Compute MSE given a PSNR (we assume the maximum pixel value is 1)."""
  return np.exp(-0.1 * np.log(10.) * psnr)

#----------------------------------------------------------------------------
# Displacement texture lookup
#----------------------------------------------------------------------------

def get_miplevels(texture: np.ndarray) -> float:
    minDim = min(texture.shape[0], texture.shape[1])
    return np.floor(np.log2(minDim))

def tex_2d(tex_map : jt.Var, coords : jt.Var, filter='nearest') -> jt.Var:
    tex_map = tex_map[None, ...]    # Add batch dimension
    tex_map = tex_map.permute(0, 3, 1, 2) # NHWC -> NCHW
    tex = F.grid_sample(tex_map, coords[None, None, ...] * 2 - 1, mode=filter, align_corners=False)
    tex = tex.permute(0, 2, 3, 1) # NCHW -> NHWC
    return tex[0, 0, ...]

#----------------------------------------------------------------------------
# Jittor texture sampling (replaces nvdiffrast.torch dr.texture)
#----------------------------------------------------------------------------

def _texture_2d_sample_jt(tex, uvs, filter_mode='linear'):
    """Sample 2D texture with UV coordinates using F.grid_sample.

    Args:
        tex: [1, H, W, C] texture in NHWC format
        uvs: [1, H', W', 2] UV coordinates in [0, 1] range
        filter_mode: 'linear' or 'nearest'
    Returns:
        [1, H', W', C] sampled texture
    """
    # NHWC -> NCHW for grid_sample
    tex_nchw = tex.permute(0, 3, 1, 2)
    # UV [0,1] -> [-1,1] for grid_sample
    grid = uvs * 2.0 - 1.0
    mode = 'bilinear' if filter_mode == 'linear' else 'nearest'
    sampled = F.grid_sample(tex_nchw, grid, mode=mode, padding_mode='border', align_corners=False)
    # NCHW -> NHWC
    return sampled.permute(0, 2, 3, 1)


def _cubemap_sample_jt(cm, dirs):
    """Sample cubemap using 3D direction vectors (replaces dr.texture boundary_mode='cube').

    Projects each direction to the appropriate cube face and performs bilinear sampling.

    Args:
        cm: [1, 6, H, W, C] cubemap in N(F) H W C format
        dirs: [1, H', W', 3] normalized direction vectors
    Returns:
        [1, H', W', C] sampled colors
    """
    B = dirs.shape[0]
    H_out, W_out = dirs.shape[1], dirs.shape[2]
    C = cm.shape[-1]
    res = cm.shape[2]  # cube face resolution

    # Normalize directions
    d = dirs / (jt.norm(dirs, p=2, dim=-1, keepdim=True) + 1e-10)
    dx, dy, dz = d[..., 0], d[..., 1], d[..., 2]

    result = jt.zeros((B, H_out, W_out, C), dtype=cm.dtype)

    for s in range(6):
        if s == 0:   # +X
            u = -dz / (dx + 1e-10)
            v = -dy / (dx + 1e-10)
            in_face = (dx >= jt.abs(dy)) & (dx >= jt.abs(dz))
        elif s == 1: # -X
            u =  dz / (-dx + 1e-10)
            v = -dy / (-dx + 1e-10)
            in_face = (-dx >= jt.abs(dy)) & (-dx >= jt.abs(dz))
        elif s == 2: # +Y
            u =  dx / (dy + 1e-10)
            v =  dz / (dy + 1e-10)
            in_face = (dy >= jt.abs(dx)) & (dy >= jt.abs(dz))
        elif s == 3: # -Y
            u =  dx / (-dy + 1e-10)
            v = -dz / (-dy + 1e-10)
            in_face = (-dy >= jt.abs(dx)) & (-dy >= jt.abs(dz))
        elif s == 4: # +Z
            u =  dx / (dz + 1e-10)
            v = -dy / (dz + 1e-10)
            in_face = (dz >= jt.abs(dx)) & (dz >= jt.abs(dy))
        else:        # -Z (s == 5)
            u = -dx / (-dz + 1e-10)
            v = -dy / (-dz + 1e-10)
            in_face = (-dz >= jt.abs(dx)) & (-dz >= jt.abs(dy))

        # Map UV from [-1, 1] to [0, 1]
        u_norm = (u * 0.5 + 0.5)
        v_norm = (v * 0.5 + 0.5)

        # Build grid for grid_sample: [B, H', W', 2] -> [-1, 1]
        grid_uv = jt.stack([u_norm, v_norm], dim=-1)

        # Sample this face
        face_tex = cm[0, s:s+1, ...]  # [1, H, W, C]
        face_nchw = face_tex.permute(0, 3, 1, 2)  # [1, C, H, W]
        grid = (grid_uv * 2.0 - 1.0).unsqueeze(0)  # [1, H', W', 2]

        sampled = F.grid_sample(face_nchw, grid, mode='bilinear',
                                padding_mode='border', align_corners=False)
        sampled = sampled.permute(0, 2, 3, 1)  # [1, H', W', C]

        # Mask and accumulate
        mask = in_face.float().unsqueeze(0).unsqueeze(-1)  # [1, H', W', 1]
        result = result + sampled * mask

    return result


#----------------------------------------------------------------------------
# Cubemap utility functions
#----------------------------------------------------------------------------

def cube_to_dir(s, x, y):
    if s == 0:   rx, ry, rz = jt.ones_like(x), -y, -x
    elif s == 1: rx, ry, rz = -jt.ones_like(x), -y, x
    elif s == 2: rx, ry, rz = x, jt.ones_like(x), y
    elif s == 3: rx, ry, rz = x, -jt.ones_like(x), -y
    elif s == 4: rx, ry, rz = x, -y, jt.ones_like(x)
    elif s == 5: rx, ry, rz = -x, -y, -jt.ones_like(x)
    return jt.stack((rx, ry, rz), dim=-1)

def latlong_to_cubemap(latlong_map, res):
    cubemap = jt.zeros(6, res[0], res[1], latlong_map.shape[-1], dtype=jt.float32)
    for s in range(6):
        gy, gx = jt.meshgrid(jt.linspace(-1.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0]),
                                jt.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1]),
                                # indexing='ij')
                                )
        v = safe_normalize(cube_to_dir(s, gx, gy))

        tu = jt.atan2(v[..., 0:1], -v[..., 2:3]) / (2 * np.pi) + 0.5
        tv = jt.acos(jt.clamp(v[..., 1:2], -1.0, 1.0)) / np.pi
        texcoord = jt.concat((tu, tv), dim=-1)

        # Jittor replacement for dr.texture(latlong_map, texcoord, filter_mode='linear')
        cubemap[s, ...] = _texture_2d_sample_jt(latlong_map[None, ...], texcoord[None, ...])[0]
    return cubemap

def cubemap_to_latlong(cubemap, res):
    gy, gx = jt.meshgrid(jt.linspace( 0.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0]),
                            jt.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1]),
                            # indexing='ij')
                            )

    sintheta, costheta = jt.sin(gy*np.pi), jt.cos(gy*np.pi)
    sinphi, cosphi     = jt.sin(gx*np.pi), jt.cos(gx*np.pi)

    reflvec = jt.stack((
        sintheta*sinphi,
        costheta,
        -sintheta*cosphi
        ), dim=-1)
    # Jittor replacement for dr.texture(cubemap, reflvec, filter_mode='linear', boundary_mode='cube')
    return _cubemap_sample_jt(cubemap[None, ...], reflvec[None, ...])[0]

def cubemap_to_latlong2(cubemap, res):
    gy, gx = jt.meshgrid(jt.linspace( 0.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0]),
                            jt.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1]),
                            # indexing='ij')
                            )

    sintheta, costheta = jt.sin(gx*np.pi), jt.cos(gx*np.pi)
    sinphi, cosphi     = jt.sin(gy*np.pi), jt.cos(gy*np.pi)

    reflvec = jt.stack((
        sintheta*sinphi,
        costheta,
        -sintheta*cosphi
        ), dim=-1)
    # Jittor replacement for dr.texture(cubemap, reflvec, filter_mode='linear', boundary_mode='cube')
    return _cubemap_sample_jt(cubemap[None, ...], reflvec[None, ...])[0]

#----------------------------------------------------------------------------
# Image scaling
#----------------------------------------------------------------------------

def scale_img_hwc(x : jt.Var, size, mag='bilinear', min='area') -> jt.Var:
    return scale_img_nhwc(x[None, ...], size, mag, min)[0]

def scale_img_nhwc(x  : jt.Var, size, mag='bilinear', min='area') -> jt.Var:
    assert (x.shape[1] >= size[0] and x.shape[2] >= size[1]) or (x.shape[1] < size[0] and x.shape[2] < size[1]), "Trying to magnify image in one dimension and minify in the other"
    y = x.permute(0, 3, 1, 2) # NHWC -> NCHW
    if x.shape[1] > size[0] and x.shape[2] > size[1]: # Minification, previous size was bigger
        y = F.interpolate(y, size, mode=min)
    else: # Magnification
        if mag == 'bilinear' or mag == 'bicubic':
            y = F.interpolate(y, size, mode=mag, align_corners=True)
        else:
            y = F.interpolate(y, size, mode=mag)
    return y.permute(0, 2, 3, 1).contiguous() # NCHW -> NHWC

def avg_pool_nhwc(x  : jt.Var, size) -> jt.Var:
    y = x.permute(0, 3, 1, 2) # NHWC -> NCHW
    y = F.avg_pool2d(y, size)
    return y.permute(0, 2, 3, 1).contiguous() # NCHW -> NHWC

#----------------------------------------------------------------------------
# Behaves similar to tf.segment_sum
#----------------------------------------------------------------------------

def segment_sum(data: jt.Var, segment_ids: jt.Var) -> jt.Var:
    num_segments = jt.unique_consecutive(segment_ids).shape[0]  # TODO: verify jt.unique_consecutive exists; if not, implement manually

    # Repeats ids until same dimension as data
    if len(segment_ids.shape) == 1:
        s = jt.prod(jt.array(data.shape[1:], dtype=jt.int64)).long()
        segment_ids = segment_ids.repeat_interleave(s).view(segment_ids.shape[0], *data.shape[1:])

    assert data.shape == segment_ids.shape, "data.shape and segment_ids.shape should be equal"

    shape = [num_segments] + list(data.shape[1:])
    result = jt.zeros(*shape, dtype=jt.float32)
    result = result.scatter_add(0, segment_ids, data)
    return result

#----------------------------------------------------------------------------
# Matrix helpers.
#----------------------------------------------------------------------------

def fovx_to_fovy(fovx, aspect):
    return np.arctan(np.tan(fovx / 2) / aspect) * 2.0

def focal_length_to_fovy(focal_length, sensor_height):
    return 2 * np.arctan(0.5 * sensor_height / focal_length)

# Reworked so this matches gluPerspective / glm::perspective, using fovy
def perspective(fovy=0.7854, aspect=1.0, n=0.1, f=1000.0):
    y = np.tan(fovy / 2)
    return jt.array([[1/(y*aspect),    0,            0,              0], 
                         [           0, 1/-y,            0,              0], 
                         [           0,    0, -(f+n)/(f-n), -(2*f*n)/(f-n)], 
                         [           0,    0,           -1,              0]], dtype=jt.float32)

# Reworked so this matches gluPerspective / glm::perspective, using fovy
def perspective_offcenter(fovy, fraction, rx, ry, aspect=1.0, n=0.1, f=1000.0):
    y = np.tan(fovy / 2)

    # Full frustum
    R, L = aspect*y, -aspect*y
    T, B = y, -y

    # Create a randomized sub-frustum
    width  = (R-L)*fraction
    height = (T-B)*fraction
    xstart = (R-L)*rx
    ystart = (T-B)*ry

    l = L + xstart
    r = l + width
    b = B + ystart
    t = b + height
    
    # https://www.scratchapixel.com/lessons/3d-basic-rendering/perspective-and-orthographic-projection-matrix/opengl-perspective-projection-matrix
    return jt.array([[2/(r-l),        0,  (r+l)/(r-l),              0], 
                         [      0, -2/(t-b),  (t+b)/(t-b),              0], 
                         [      0,        0, -(f+n)/(f-n), -(2*f*n)/(f-n)], 
                         [      0,        0,           -1,              0]], dtype=jt.float32)

def translate(x, y, z):
    return jt.array([[1, 0, 0, x], 
                         [0, 1, 0, y], 
                         [0, 0, 1, z], 
                         [0, 0, 0, 1]], dtype=jt.float32)

def rotate_x(a):
    s, c = np.sin(a), np.cos(a)
    return jt.array([[1,  0, 0, 0], 
                         [0,  c, s, 0], 
                         [0, -s, c, 0], 
                         [0,  0, 0, 1]], dtype=jt.float32)

def rotate_y(a):
    s, c = np.sin(a), np.cos(a)
    return jt.array([[ c, 0, s, 0], 
                         [ 0, 1, 0, 0], 
                         [-s, 0, c, 0], 
                         [ 0, 0, 0, 1]], dtype=jt.float32)

def scale(s):
    return jt.array([[ s, 0, 0, 0], 
                         [ 0, s, 0, 0], 
                         [ 0, 0, s, 0], 
                         [ 0, 0, 0, 1]], dtype=jt.float32)

def lookAt(eye, at, up):
    a = eye - at
    w = a / jt.norm(a)
    u = jt.cross(up, w)
    u = u / jt.norm(u)
    v = jt.cross(w, u)
    translate = jt.array([[1, 0, 0, -eye[0]], 
                              [0, 1, 0, -eye[1]], 
                              [0, 0, 1, -eye[2]], 
                              [0, 0, 0, 1]], dtype=eye.dtype)
    rotate = jt.array([[u[0], u[1], u[2], 0], 
                           [v[0], v[1], v[2], 0], 
                           [w[0], w[1], w[2], 0], 
                           [0, 0, 0, 1]], dtype=eye.dtype)
    return rotate @ translate

@jt.no_grad()
def random_rotation_translation(t):
    m = np.random.normal(size=[3, 3])
    m[1] = np.cross(m[0], m[2])
    m[2] = np.cross(m[0], m[1])
    m = m / np.linalg.norm(m, axis=1, keepdims=True)
    m = np.pad(m, [[0, 1], [0, 1]], mode='constant')
    m[3, 3] = 1.0
    m[:3, 3] = np.random.uniform(-t, t, size=[3])
    return jt.array(m, dtype=jt.float32)

@jt.no_grad()
def random_rotation(device=None):
    m = np.random.normal(size=[3, 3])
    m[1] = np.cross(m[0], m[2])
    m[2] = np.cross(m[0], m[1])
    m = m / np.linalg.norm(m, axis=1, keepdims=True)
    m = np.pad(m, [[0, 1], [0, 1]], mode='constant')
    m[3, 3] = 1.0
    m[:3, 3] = np.array([0,0,0]).astype(np.float32)
    return jt.array(m, dtype=jt.float32)

#----------------------------------------------------------------------------
# Compute focal points of a set of lines using least squares. 
# handy for poorly centered datasets
#----------------------------------------------------------------------------

def lines_focal(o, d):
    d = safe_normalize(d)
    # jt.eye not available in Jittor — construct manually
    I = jt.zeros(3, 3, dtype=o.dtype)
    I[0, 0] = I[1, 1] = I[2, 2] = 1.0
    S = jt.sum(d[..., None] @ jt.transpose(d[..., None], 1, 2) - I[None, ...], dim=0)
    C = jt.sum((d[..., None] @ jt.transpose(d[..., None], 1, 2) - I[None, ...]) @ o[..., None], dim=0).squeeze(1)
    return jt.array(np.linalg.pinv(S.numpy())) @ C

#----------------------------------------------------------------------------
# Cosine sample around a vector N
#----------------------------------------------------------------------------
@jt.no_grad()
def cosine_sample(N, size=None):
    # construct local frame
    N = N/jt.norm(N)

    dx0 = jt.array([0, N[2], -N[1]], dtype=N.dtype)
    dx1 = jt.array([-N[2], 0, N[0]], dtype=N.dtype)

    mask = (dot(dx0, dx0) > dot(dx1, dx1)).float()
    dx = mask * dx0 + (1.0 - mask) * dx1
    #dx = dx0 if np.dot(dx0,dx0) > np.dot(dx1,dx1) else dx1
    dx = dx / jt.norm(dx)
    dy = jt.cross(N,dx)
    dy = dy / jt.norm(dy)

    # cosine sampling in local frame
    if size is None:
        phi = 2.0 * np.pi * np.random.uniform()
        s = np.random.uniform()
    else:
        phi = 2.0 * np.pi * jt.rand(*size, 1, dtype=N.dtype)
        s = jt.rand(*size, 1, dtype=N.dtype)
    costheta = np.sqrt(s)
    sintheta = np.sqrt(1.0 - s)

    # cartesian vector in local space
    x = np.cos(phi)*sintheta
    y = np.sin(phi)*sintheta
    z = costheta

    # local to world
    return dx*x + dy*y + N*z

#----------------------------------------------------------------------------
# Bilinear downsample by 2x.
#----------------------------------------------------------------------------

def bilinear_downsample(x : jt.Var) -> jt.Var:
    w = jt.array([[1, 3, 3, 1], [3, 9, 9, 3], [3, 9, 9, 3], [1, 3, 3, 1]], dtype=jt.float32) / 64.0
    w = w.expand(x.shape[-1], 1, 4, 4) 
    x = F.conv2d(x.permute(0, 3, 1, 2), w, padding=1, stride=2, groups=x.shape[-1])
    return x.permute(0, 2, 3, 1)

#----------------------------------------------------------------------------
# Bilinear downsample log(spp) steps
#----------------------------------------------------------------------------

def bilinear_downsample(x : jt.Var, spp) -> jt.Var:
    w = jt.array([[1, 3, 3, 1], [3, 9, 9, 3], [3, 9, 9, 3], [1, 3, 3, 1]], dtype=jt.float32) / 64.0
    g = x.shape[-1]
    w = w.expand(g, 1, 4, 4) 
    x = x.permute(0, 3, 1, 2) # NHWC -> NCHW
    steps = int(np.log2(spp))
    for _ in range(steps):
        xp = F.pad(x, (1,1,1,1), mode='replicate')
        x = F.conv2d(xp, w, padding=0, stride=2, groups=g)
    return x.permute(0, 2, 3, 1).contiguous() # NCHW -> NHWC

#----------------------------------------------------------------------------
# Singleton initialize GLFW
#----------------------------------------------------------------------------

_glfw_initialized = False
def init_glfw():
    global _glfw_initialized
    try:
        import glfw
        glfw.ERROR_REPORTING = 'raise'
        glfw.default_window_hints()
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        test = glfw.create_window(8, 8, "Test", None, None) # Create a window and see if not initialized yet
    except glfw.GLFWError as e:
        if e.error_code == glfw.NOT_INITIALIZED:
            glfw.init()
            _glfw_initialized = True

#----------------------------------------------------------------------------
# Image display function using OpenGL.
#----------------------------------------------------------------------------

_glfw_window = None
def display_image(image, title=None):
    # Import OpenGL
    import OpenGL.GL as gl
    import glfw

    # Zoom image if requested.
    image = np.asarray(image[..., 0:3]) if image.shape[-1] == 4 else np.asarray(image)
    height, width, channels = image.shape

    # Initialize window.
    init_glfw()
    if title is None:
        title = 'Debug window'
    global _glfw_window
    if _glfw_window is None:
        glfw.default_window_hints()
        _glfw_window = glfw.create_window(width, height, title, None, None)
        glfw.make_context_current(_glfw_window)
        glfw.show_window(_glfw_window)
        glfw.swap_interval(0)
    else:
        glfw.make_context_current(_glfw_window)
        glfw.set_window_title(_glfw_window, title)
        glfw.set_window_size(_glfw_window, width, height)

    # Update window.
    glfw.poll_events()
    gl.glClearColor(0, 0, 0, 1)
    gl.glClear(gl.GL_COLOR_BUFFER_BIT)
    gl.glWindowPos2f(0, 0)
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
    gl_format = {3: gl.GL_RGB, 2: gl.GL_RG, 1: gl.GL_LUMINANCE}[channels]
    gl_dtype = {'uint8': gl.GL_UNSIGNED_BYTE, 'float32': gl.GL_FLOAT}[image.dtype.name]
    gl.glDrawPixels(width, height, gl_format, gl_dtype, image[::-1])
    glfw.swap_buffers(_glfw_window)
    if glfw.window_should_close(_glfw_window):
        return False
    return True

#----------------------------------------------------------------------------
# Image save/load helper.
#----------------------------------------------------------------------------

def save_image(fn, x : np.ndarray):
    try:
        if os.path.splitext(fn)[1] == ".png":
            imageio.imwrite(fn, np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8), compress_level=3) # Low compression for faster saving
        else:
            imageio.imwrite(fn, np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8))
    except:
        print("WARNING: FAILED to save image %s" % fn)

def save_image_raw(fn, x : np.ndarray):
    try:
        imageio.imwrite(fn, x)
    except:
        print("WARNING: FAILED to save image %s" % fn)


def load_image_raw(fn) -> np.ndarray:
    return imageio.imread(fn)


def load_image(fn) -> np.ndarray:
    img = load_image_raw(fn)
    if img.dtype == np.float32: # HDR image
        return img
    else: # LDR image
        return img.astype(np.float32) / 255

#----------------------------------------------------------------------------

def time_to_text(x):
    if x > 3600:
        return "%.2f h" % (x / 3600)
    elif x > 60:
        return "%.2f m" % (x / 60)
    else:
        return "%.2f s" % x

#----------------------------------------------------------------------------

def checkerboard(res, checker_size) -> np.ndarray:
    tiles_y = (res[0] + (checker_size*2) - 1) // (checker_size*2)
    tiles_x = (res[1] + (checker_size*2) - 1) // (checker_size*2)
    check = np.kron([[1, 0] * tiles_x, [0, 1] * tiles_x] * tiles_y, np.ones((checker_size, checker_size)))*0.33 + 0.33
    check = check[:res[0], :res[1]]
    return np.stack((check, check, check), axis=-1)


