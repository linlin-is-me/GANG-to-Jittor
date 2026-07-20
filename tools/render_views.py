"""
Standalone renderer: load a GANG checkpoint and render all test views.

Usage:
    python tools/render_views.py \
        --npz outputs/treehill_p44_7k/chkpnt7000.npz \
        --out_dir outputs/treehill_p44_7k/renders

Output:
    outputs/treehill_p44_7k/renders/
        view_000.png   ...  view_017.png   (render)
        view_000_gt.png ...  view_017_gt.png (ground truth)
"""
import sys, os, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'submodules'))
import jittor as jt; jt.flags.use_cuda = 1
from argparse import Namespace
from tqdm import tqdm

from scene.gaussian_model import GaussianModel
from scene import Scene
from gaussian_renderer import render, prefilter_voxel
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz', required=True, help='Checkpoint file (.npz)')
    parser.add_argument('--out_dir', required=True, help='Output directory for rendered images')
    parser.add_argument('--source_path', default='data/treehill')
    parser.add_argument('--resolution', type=int, default=8)
    parser.add_argument('--is_pbr', type=int, default=0)
    parser.add_argument('--max_points', type=int, default=-1)
    parser.add_argument('--start_idx', type=int, default=0, help='First camera index to render')
    parser.add_argument('--end_idx', type=int, default=-1, help='Last camera index (exclusive, -1 = all)')
    args = parser.parse_args()

    is_pbr = bool(args.is_pbr)
    npz_path = args.npz
    out_dir = args.out_dir
    os.makedirs(os.path.join(out_dir, 'render'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'gt'), exist_ok=True)

    if not os.path.exists(npz_path):
        print(f"ERROR: checkpoint not found: {npz_path}")
        sys.exit(1)

    # ---- Build model & load checkpoint ----
    lp = Namespace(
        source_path=args.source_path, model_path=os.path.dirname(npz_path),
        images='images', resolution=args.resolution, eval=True,
        resolution_scales=[1.0], data_device='cuda',
        feat_dim=32, n_offsets=10, fork=2, use_feat_bank=False, appearance_dim=0,
        add_opacity_dist=False, add_cov_dist=False, add_color_dist=False, add_level=False,
        visible_threshold=0.1, dist2level='round', base_layer=10, progressive=True,
        extend=1.1, is_pbr=is_pbr, normal_detal=False, with_meta=True,
        bound=1.5, ratio=1, ds=1, undistorted=False, max_points=args.max_points,
        dist_ratio=0.999, init_level=-1, levels=-1,
        white_background=False, random_background=False
    )
    g = GaussianModel(lp.feat_dim, lp.n_offsets, lp.fork, lp.use_feat_bank, lp.appearance_dim,
        lp.add_opacity_dist, lp.add_cov_dist, lp.add_color_dist,
        lp.add_level, lp.visible_threshold, lp.dist2level,
        lp.base_layer, lp.progressive, lp.extend, lp.is_pbr, lp.normal_detal, lp.with_meta)

    print(f"Loading checkpoint: {npz_path}")
    light_state = None
    if npz_path.endswith('.pkl'):
        import pickle
        data = pickle.load(open(npz_path, 'rb'))
        np_data = data['data']  # list of 19 items
        saved_iter = data['iteration']
    else:
        raw = dict(np.load(npz_path, allow_pickle=True))
        saved_iter = int(raw.pop('iteration', 0))
        n_items = raw.pop('_n_items', None)
        if n_items is None:
            n_items = max(int(k.split('_')[1]) for k in raw if k.startswith('item_')) + 1
        else:
            n_items = int(n_items)
        np_data = [raw.get(f'item_{i}') for i in range(n_items)]
        # Phase 64: extract light state from raw dict if present
        if 'item_0' not in raw:  # raw only has non-item keys after pop
            pass  # light keys were already extracted via np_data above
    light_state = g.restore_numpy(np_data)
    print(f"  Restored iter {saved_iter}, anchors={g.get_anchor.shape[0]}")

    # Setup optimizer (needed for model internals)
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
    g.optimizer = Namespace(param_groups=l)
    g.optimizer.state_dict = lambda: {}
    g.optimizer.load_state_dict = lambda d: None
    N = g.get_anchor.shape[0]
    g.opacity_accum = jt.zeros((N, 1))
    g.offset_gradient_accum = jt.zeros((N * K, 1))
    g.offset_denom = jt.zeros((N * K, 1))
    g.anchor_demon = jt.zeros((N, 1))

    # Build scene (cameras only, skip octree rebuild)
    # NOTE: model trained at a specific resolution; rendering at a different
    # resolution requires proper LOD recalculation (Phase 34, pending).
    # For now, render at training resolution (resolution=16) for correct output.
    s = Scene(lp, g, shuffle=False, resolution_scales=lp.resolution_scales, is_pbr=is_pbr,
              skip_octree=True)
    g.eval()  # Phase 78: set MLPs to eval mode
    test_cams = s.getTestCameras()
    print(f"Test cameras: {len(test_cams)}")

    # PBR cubemap
    light = None
    if is_pbr:
        from scene.NVDIFFREC.light import Hybridlight
        light = Hybridlight(base_res=256, num_sg=16, cache_dir=os.path.dirname(npz_path))
        if light_state is not None:
            light.load_from_numpy(light_state)
        print("  PBR cubemap loaded")

    pipe = Namespace(compute_cov3D_python=False, debug=False, sample_num=64)
    bg = jt.float32([0, 0, 0])

    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx > 0 else len(test_cams)

    for idx in range(start_idx, end_idx):
        cam = test_cams[idx]
        print(f"Rendering view {idx}/{end_idx}...", end=' ', flush=True)

        try: g.set_anchor_mask(cam.camera_center, 99999, cam.resolution_scale)
        except: pass
        voxel_mask = prefilter_voxel(cam, g, pipe, bg)
        render_pkg = render(cam, g, pipe, bg, visible_mask=voxel_mask,
                            is_pbr=is_pbr, light=light, is_training=False)

        image = render_pkg["render"].float32()
        gt = cam.original_image.float32()

        # Save render
        img_np = image.clamp(0, 1).numpy().transpose(1, 2, 0)
        img_np = (img_np * 255).astype(np.uint8)
        Image.fromarray(img_np).save(os.path.join(out_dir, 'render', f'view_{idx:03d}.png'))

        # Save GT
        gt_np = gt.clamp(0, 1).numpy().transpose(1, 2, 0)
        gt_np = (gt_np * 255).astype(np.uint8)
        Image.fromarray(gt_np).save(os.path.join(out_dir, 'gt', f'view_{idx:03d}_gt.png'))

        # Cleanup GPU memory between views (critical for res=1 PBR)
        del render_pkg, image, gt, voxel_mask
        jt.sync_all()
        jt.gc()
        jt.gc()

        print("done")

    print(f"\nDone. {end_idx - start_idx} views saved to {out_dir}/")


if __name__ == '__main__':
    main()
