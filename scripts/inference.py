"""
inference.py — Per-scene interactive inference
===============================================
Usage:
    PYTHONPATH=$PWD/src python scripts/inference.py \
        --data_root ~/Downloads/dl_challenge --weights best_model.pth

Outputs interactive Plotly HTML files (one per scene) with GT vs predicted
3D boxes overlaid on the point cloud.
"""

import os, argparse
import cv2, torch
import numpy as np
import plotly.graph_objects as go

from sereact_bbox.model  import DGCNNBBox
from sereact_bbox.config import DataConfig, ModelConfig, DEFAULT_OUTPUT_DIR, DEFAULT_CHECKPOINT
from sereact_bbox.dataset import _mad_filter   # reuse the same filter as training

EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]


def add_box(fig, corners, color, name):
    xl, yl, zl = [], [], []
    for i, j in EDGES:
        xl += [corners[i,0], corners[j,0], None]
        yl += [corners[i,1], corners[j,1], None]
        zl += [corners[i,2], corners[j,2], None]
    fig.add_trace(go.Scatter3d(x=xl, y=yl, z=zl, mode="lines",
                               line=dict(color=color, width=4), name=name))


def preprocess_instance(pc, img_rgb, mask, num_points):
    """Identical contextual sampling to training"""
    obj_y, obj_x = np.where(mask)
    bg_y,  bg_x  = np.where(~mask)
    if len(obj_y) == 0: return None

    pc_obj  = pc[:, obj_y, obj_x]
    rgb_obj = img_rgb[obj_y, obj_x].astype(np.float32).T / 255.0
    pc_bg   = pc[:, bg_y, bg_x]
    rgb_bg  = img_rgb[bg_y, bg_x].astype(np.float32).T / 255.0

    # Clean depth
    valid_obj = pc_obj[2] > DataConfig.depth_threshold
    pc_obj, rgb_obj = pc_obj[:, valid_obj], rgb_obj[:, valid_obj]
    valid_bg  = pc_bg[2]  > DataConfig.depth_threshold
    pc_bg,  rgb_bg  = pc_bg[:, valid_bg],  rgb_bg[:, valid_bg]

    # Clean MAD outliers
    pc_obj, rgb_obj = _mad_filter(pc_obj, rgb_obj)
    pc_bg,  rgb_bg  = _mad_filter(pc_bg,  rgb_bg)

    n_obj = num_points // 2
    n_bg  = num_points - n_obj
    N_obj, N_bg = pc_obj.shape[1], pc_bg.shape[1]

    if N_obj > 0:
        c_obj = np.random.choice(N_obj, n_obj, replace=(N_obj < n_obj))
        pc_obj, rgb_obj = pc_obj[:, c_obj], rgb_obj[:, c_obj]
    else:
        pc_obj, rgb_obj = np.zeros((3, n_obj), dtype=np.float32), np.zeros((3, n_obj), dtype=np.float32)

    if N_bg > 0:
        c_bg = np.random.choice(N_bg, n_bg, replace=(N_bg < n_bg))
        pc_bg, rgb_bg = pc_bg[:, c_bg], rgb_bg[:, c_bg]
    else:
        pc_bg, rgb_bg = np.zeros((3, n_bg), dtype=np.float32), np.zeros((3, n_bg), dtype=np.float32)

    mask_obj = np.ones((1, n_obj),  dtype=np.float32)
    mask_bg  = np.zeros((1, n_bg), dtype=np.float32)

    pc_c  = np.concatenate([pc_obj, pc_bg], axis=1)
    rgb_c = np.concatenate([rgb_obj, rgb_bg], axis=1)
    m_c   = np.concatenate([mask_obj, mask_bg], axis=1)

    anchor = np.median(pc_obj, axis=1) if N_obj > 0 else np.zeros(3, dtype=np.float32)
    pc_c   = pc_c - anchor.reshape(3,1)

    feat = np.concatenate([pc_c, rgb_c, m_c], axis=0)
    return feat, anchor


def run_sample(model, device, sample_dir, out_dir, idx, total, n_pts):
    img_bgr = cv2.imread(os.path.join(sample_dir, "rgb.jpg"))
    if img_bgr is None: return
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pc      = np.load(os.path.join(sample_dir, "pc.npy"), allow_pickle=True).astype(np.float32)
    masks   = np.load(os.path.join(sample_dir, "mask.npy"), allow_pickle=True)

    folder = os.path.basename(sample_dir.rstrip("/"))
    print(f"[{idx}/{total}] {folder} — {len(masks)} object(s)")

    pred_boxes = []
    for i, mask in enumerate(masks):
        res = preprocess_instance(pc, img_rgb, mask, n_pts)
        if res is None: continue
        feat, anchor = res
        inp = torch.from_numpy(feat).unsqueeze(0).to(device)

        with torch.no_grad():
            center, log_dims, rot6d = model(inp)
            corners = model.get_3d_box(center, log_dims, rot6d)

        corners_abs = corners[0].cpu().numpy() + anchor
        pred_boxes.append(corners_abs)
        dims = torch.exp(log_dims[0]).cpu().numpy()
        print(f"  Obj {i+1}: {dims[0]:.3f}m × {dims[1]:.3f}m × {dims[2]:.3f}m")

    fig = go.Figure()

    pc_flat = pc.reshape(3, -1)
    valid   = pc_flat[2] > 0.01
    pc_vis  = pc_flat[:, valid]

    n_sub = max(1, pc_vis.shape[1] // 100)
    sub   = np.random.choice(pc_vis.shape[1], n_sub, replace=False)
    pc_sub = pc_vis[:, sub]

    fig.add_trace(go.Scatter3d(x=pc_sub[0], y=pc_sub[1], z=pc_sub[2],
                               mode='markers',
                               marker=dict(size=1.5, color=pc_sub[2], colorscale='Viridis', opacity=0.5),
                               name='Point Cloud'))

    for i, box in enumerate(pred_boxes):
        add_box(fig, box, 'red', f'Pred Obj {i+1}')

    gt_path = os.path.join(sample_dir, "bbox3d.npy")
    if os.path.exists(gt_path):
        gt_boxes = np.load(gt_path, allow_pickle=True)
        for i, box in enumerate(gt_boxes):
            add_box(fig, box, 'green', f'GT Obj {i+1}')

    fig.update_layout(title=f"3D Bounding Boxes: {folder}",
                      scene=dict(aspectmode="data"),
                      margin=dict(l=0,r=0,b=0,t=40))
    out = os.path.join(out_dir, f"{folder}_interactive.html")
    fig.write_html(out)
    print(f"  Saved: {out}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = DGCNNBBox(in_channels=ModelConfig.in_channels).to(device)
    if os.path.exists(args.weights):
        # FIX: weights_only=False added for PyTorch 2.6+
        ckpt  = torch.load(args.weights, map_location=device, weights_only=False)
        state = {k.replace("_orig_mod.",""):v for k,v in ckpt.get("model",ckpt).items()}
        model.load_state_dict(state, strict=False)
        print(f"Loaded: epoch={ckpt.get('epoch','?')} best_mcd={ckpt.get('best_mcd',float('nan')):.4f}m")
    model.eval()
    os.makedirs(args.out_dir, exist_ok=True)

    samples = [args.sample] if args.sample else sorted(
        p for p in (os.path.join(args.data_root,d) for d in os.listdir(args.data_root))
        if os.path.isdir(p) and os.path.exists(os.path.join(p,"rgb.jpg")))

    for i, sd in enumerate(samples, 1):
        run_sample(model, device, sd, args.out_dir, i, len(samples), DataConfig.num_points)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights",   default=DEFAULT_CHECKPOINT)
    p.add_argument("--out_dir",   default=DEFAULT_OUTPUT_DIR)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--sample",    type=str)
    g.add_argument("--data_root", type=str)
    main(p.parse_args())