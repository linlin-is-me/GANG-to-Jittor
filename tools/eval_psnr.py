"""
Evaluate PSNR/SSIM of a trained model on test views.

Usage:
    python tools/eval_psnr.py --model_path outputs/treehill_10k_m30 --iteration 10000
    python tools/eval_psnr.py --model_path outputs/treehill_planD_full --ply_only
"""
import sys, os, time, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'submodules'))
import jittor as jt; jt.flags.use_cuda = 1
from argparse import Namespace
from tqdm import tqdm

from scene.gaussian_model import GaussianModel
from scene import Scene
from gaussian_renderer import render, prefilter_voxel
from utils.image_utils import psnr
from utils.loss_utils import ssim
from utils.general_utils import get_expon_lr_func


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--iteration', type=int, default=-1, help='Checkpoint iteration (default: latest)')
    parser.add_argument('--source_path', default='data/treehill')
    parser.add_argument('--resolution', type=int, default=-1)
    parser.add_argument('--is_pbr', type=int, default=0)
    parser.add_argument('--save_images', type=int, default=1, help='Save rendered images')
    args = parser.parse_args()

    is_pbr = bool(args.is_pbr)
    model_path = args.model_path
    os.makedirs(os.path.join(model_path, 'eval'), exist_ok=True)

    # Find checkpoint
    if args.iteration > 0:
        npz_path = os.path.join(model_path, f"chkpnt{args.iteration}.npz")
    else:
        # Find latest checkpoint
        import glob
        ckpts = sorted(glob.glob(os.path.join(model_path, "chkpnt*.npz")))
        if not ckpts:
            print("No checkpoint found. Trying PLY...")
            ckpts = []
        npz_path = ckpts[-1] if ckpts else None

    # Build model params
    lp = Namespace(
        source_path=args.source_path, model_path=model_path, images='images',
        resolution=args.resolution, eval=True, resolution_scales=[1.0], data_device='cuda',
        feat_dim=32, n_offsets=10, fork=2, use_feat_bank=False, appearance_dim=0,
        add_opacity_dist=False, add_cov_dist=False, add_color_dist=False, add_level=False,
        visible_threshold=0.1, dist2level='round', base_layer=10, progressive=True,
        extend=1.1, is_pbr=is_pbr, normal_detal=False, with_meta=True,
        bound=1.5, ratio=1, ds=1, undistorted=False, max_points=-1,  # -1 = all points
        dist_ratio=0.999, init_level=-1, levels=-1,
        white_background=False, random_background=False
    )
    g = GaussianModel(lp.feat_dim, lp.n_offsets, lp.fork, lp.use_feat_bank, lp.appearance_dim,
        lp.add_opacity_dist, lp.add_cov_dist, lp.add_color_dist,
        lp.add_level, lp.visible_threshold, lp.dist2level,
        lp.base_layer, lp.progressive, lp.extend, lp.is_pbr, lp.normal_detal, lp.with_meta)

    # Load checkpoint
    if npz_path and os.path.exists(npz_path):
        print(f"Loading checkpoint: {npz_path}")
        raw = dict(np.load(npz_path, allow_pickle=True))
        saved_iter = int(raw.pop('iteration', 0))
        # Use _n_items if present (new format), else guess from max item_* key
        n_items = raw.pop('_n_items', None)
        if n_items is None:
            n_items = max(int(k.split('_')[1]) for k in raw if k.startswith('item_')) + 1
        else:
            n_items = int(n_items)
        np_data = [raw.get(f'item_{i}') for i in range(n_items)]
        g.restore_numpy(np_data)
        print(f"  Restored iter {saved_iter}, anchors={g.get_anchor.shape[0]}")
        # CRITICAL: restore_numpy() already rebuilds the octree parameters
        # (standard_dist, voxel_size, levels). Do NOT call Scene() which would
        # rebuild the octree from SfM points and overwrite the loaded anchors.
        # Only build cameras (no octree rebuild).
        s = Scene(lp, g, shuffle=False, resolution_scales=lp.resolution_scales, is_pbr=is_pbr,
                  skip_octree=True)  # skip octree rebuild — use loaded anchors
    else:
        print("No .npz checkpoint found, building from PLY...")
        s = Scene(lp, g, shuffle=False, resolution_scales=lp.resolution_scales, is_pbr=is_pbr)
        ply_iter = args.iteration if args.iteration > 0 else 10000
        ply_dir = os.path.join(model_path, 'point_cloud', f'iteration_{ply_iter}')
        if os.path.exists(ply_dir):
            g.load_ply(os.path.join(ply_dir, 'point_cloud.ply'))
            g.load_mlp_checkpoints(ply_dir)
            print(f"  Loaded PLY from iter {ply_iter}")
        else:
            print(f"  PLY dir not found: {ply_dir}")
            sys.exit(1)

    # Setup lightweight optimizer (needed for model internals, not for training)
    K = g.n_offsets
    _setup_optimizer(g, is_pbr)
    g.eval()  # Phase 78: set MLPs to eval mode (avoids per-frame coarse_intervals AttributeError)
    test_cams = s.getTestCameras()
    print(f"Test cameras: {len(test_cams)}")

    # Setup cubemap for PBR
    light = None
    if is_pbr:
        from scene.NVDIFFREC.light import Hybridlight
        light = Hybridlight(dir=os.path.join(args.source_path))
        light.build_mips()
        print("  PBR cubemap built")

    pipe = Namespace(compute_cov3D_python=False, debug=False, sample_num=64)
    bg = jt.float32([0, 0, 0])

    psnr_list = []
    ssim_list = []
    t_start = time.time()

    for idx, cam in enumerate(tqdm(test_cams, desc="Evaluating")):
        try: g.set_anchor_mask(cam.camera_center, 99999, cam.resolution_scale)
        except: pass
        voxel_mask = prefilter_voxel(cam, g, pipe, bg)
        render_pkg = render(cam, g, pipe, bg, visible_mask=voxel_mask,
                            is_pbr=is_pbr, light=light, is_training=False)

        image = render_pkg["render"].float32()
        gt = cam.original_image.float32()

        p_val = float(psnr(image, gt).mean().numpy())
        s_val = float(ssim(image, gt).mean().numpy())
        psnr_list.append(p_val)
        ssim_list.append(s_val)

        # Save first 5 images
        if args.save_images and idx < 5:
            os.makedirs(os.path.join(model_path, 'eval', 'render'), exist_ok=True)
            os.makedirs(os.path.join(model_path, 'eval', 'gt'), exist_ok=True)
            try:
                _save_image(image, os.path.join(model_path, 'eval', 'render', f'{idx:03d}.png'))
                _save_image(gt, os.path.join(model_path, 'eval', 'gt', f'{idx:03d}.png'))
            except:
                pass

    elapsed = time.time() - t_start
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)

    print(f"\n{'='*60}")
    print(f"Results ({len(test_cams)} views, {elapsed:.1f}s):")
    print(f"  PSNR: {avg_psnr:.2f} dB")
    print(f"  SSIM: {avg_ssim:.4f}")
    print(f"  Best  PSNR: {max(psnr_list):.2f} dB (view {np.argmax(psnr_list)})")
    print(f"  Worst PSNR: {min(psnr_list):.2f} dB (view {np.argmin(psnr_list)})")
    print(f"{'='*60}")

    # Save results
    with open(os.path.join(model_path, 'eval', 'results.txt'), 'w') as f:
        f.write(f"PSNR: {avg_psnr:.2f} dB\n")
        f.write(f"SSIM: {avg_ssim:.4f}\n")
        f.write(f"Views: {len(test_cams)}\n")
        f.write(f"Per-view PSNR: {psnr_list}\n")
        f.write(f"Per-view SSIM: {ssim_list}\n")


def _setup_optimizer(g, is_pbr):
    """Lightweight optimizer for model internals (not training)."""
    K = g.n_offsets
    l = [
        {"params": [g._anchor], "lr": 0.0, "name": "anchor"},
        {"params": [g._offset], "lr": 0.01 * g.spatial_lr_scale, "name": "offset"},
        {"params": [g._anchor_feat], "lr": 0.0075, "name": "anchor_feat"},
        {"params": [g._opacity], "lr": 0.02, "name": "opacity"},
        {"params": [g._scaling], "lr": 0.007, "name": "scaling"},
        {"params": [g._rotation], "lr": 0.002, "name": "rotation"},
        {"params": g.mlp_opacity.parameters(), "lr": 0.002, "name": "mlp_opacity"},
        {"params": g.mlp_cov.parameters(), "lr": 0.004, "name": "mlp_cov"},
        {"params": g.mlp_color.parameters(), "lr": 0.008, "name": "mlp_color"},
    ]
    if is_pbr:
        l.extend([
            {"params": g.mlp_albedo.parameters(), "lr": 0.075, "name": "mlp_albedo"},
            {"params": g.mlp_roughness.parameters(), "lr": 0.005, "name": "mlp_roughness"},
            {"params": g.mlp_matallic.parameters(), "lr": 0.005, "name": "mlp_matallic"},
        ])
    g.optimizer = Namespace(param_groups=l)
    g.optimizer.state_dict = lambda: {}
    g.optimizer.load_state_dict = lambda d: None
    # Init accumulators (needed for set_anchor_mask)
    g.opacity_accum = jt.zeros((g.get_anchor.shape[0], 1))
    g.offset_gradient_accum = jt.zeros((g.get_anchor.shape[0] * K, 1))
    g.offset_denom = jt.zeros((g.get_anchor.shape[0] * K, 1))
    g.anchor_demon = jt.zeros((g.get_anchor.shape[0], 1))


def _save_image(tensor, path):
    """Save [3,H,W] Jittor tensor as PNG."""
    from PIL import Image
    arr = tensor.clamp(0, 1).numpy()
    arr = (arr.transpose(1, 2, 0) * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


if __name__ == '__main__':
    main()
