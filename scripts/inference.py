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
    """Identical preprocessing to dataset.__getitem__ (no augmentation)."""
    obj_y, obj_x = np.where(mask)
    bg_y,  bg_x  = np.where(~mask)
    if len(obj_y) == 0:
        return None, None

    pc_obj  = pc[:, obj_y, obj_x]
    rgb_obj = img_rgb[obj_y, obj_x].astype(np.float32).T / 255.0
    pc_bg   = pc[:, bg_y,  bg_x]
    rgb_bg  = img_rgb[bg_y,  bg_x].astype(np.float32).T / 255.0

    valid_obj = pc_obj[2] > 0.01;  pc_obj,  rgb_obj = pc_obj[:, valid_obj],  rgb_obj[:, valid_obj]
    valid_bg  = pc_bg[2]  > 0.01;  pc_bg,   rgb_bg  = pc_bg[:, valid_bg],   rgb_bg[:, valid_bg]

    # MAD filter — same as training
    pc_obj, rgb_obj = _mad_filter(pc_obj, rgb_obj)

    anchor = np.median(pc_obj, axis=1) if pc_obj.shape[1] > 0 else np.zeros(3, dtype=np.float32)

    n_obj, n_bg = num_points // 2, num_points - num_points // 2
    N_obj, N_bg = pc_obj.shape[1], pc_bg.shape[1]

    rng = np.random.default_rng(seed=42)   # deterministic at inference
    c_o = rng.choice(N_obj, n_obj, replace=(N_obj < n_obj)) if N_obj > 0 \
          else None
    c_b = rng.choice(N_bg,  n_bg,  replace=(N_bg  < n_bg))  if N_bg  > 0 \
          else None

    pc_obj  = pc_obj[:,  c_o] if c_o is not None else np.zeros((3, n_obj), dtype=np.float32)
    rgb_obj = rgb_obj[:, c_o] if c_o is not None else np.zeros((3, n_obj), dtype=np.float32)
    pc_bg   = pc_bg[:,   c_b] if c_b is not None else np.zeros((3, n_bg),  dtype=np.float32)
    rgb_bg  = rgb_bg[:,  c_b] if c_b is not None else np.zeros((3, n_bg),  dtype=np.float32)

    pc_comb  = np.concatenate([pc_obj,  pc_bg],  axis=1) - anchor[:,None]
    rgb_comb = np.concatenate([rgb_obj, rgb_bg],  axis=1)
    mask_comb= np.concatenate([np.ones((1,n_obj),dtype=np.float32),
                                np.zeros((1,n_bg), dtype=np.float32)], axis=1)
    features = np.concatenate([pc_comb, rgb_comb, mask_comb], axis=0)
    return features, anchor


def run_sample(model, device, sample_dir, out_dir, idx, total, num_points):
    img_bgr = cv2.imread(os.path.join(sample_dir, "rgb.jpg"))
    if img_bgr is None: return
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pc      = np.load(os.path.join(sample_dir, "pc.npy"),
                      allow_pickle=True).astype(np.float32)
    masks   = np.load(os.path.join(sample_dir, "mask.npy"), allow_pickle=True)

    folder  = os.path.basename(sample_dir.rstrip("/"))
    print(f"\n[{idx}/{total}] {folder} — {len(masks)} object(s)")

    predicted = []
    for i, mask in enumerate(masks):
        features, anchor = preprocess_instance(pc, img_rgb, mask, num_points)
        if features is None: continue

        inp = torch.from_numpy(features).unsqueeze(0).to(device)
        with torch.no_grad():
            center, log_dims, rot6d = model(inp)
            pred_c = model.get_3d_box(center, log_dims, rot6d)

        corners_abs = pred_c[0].cpu().numpy() + anchor
        predicted.append(corners_abs)
        dims = torch.exp(log_dims[0]).cpu().numpy()
        print(f"  Obj {i+1}: {dims[0]:.3f}m × {dims[1]:.3f}m × {dims[2]:.3f}m")

    # ── Point cloud (MAD-cleaned for scatter) ─────────────────────────────
    pc_flat = pc.reshape(3,-1)
    pc_vis  = pc_flat[:, pc_flat[2] > 0.01]
    med = np.median(pc_vis, axis=1, keepdims=True)
    mad = np.median(np.abs(pc_vis-med), axis=1, keepdims=True) + 1e-6
    inlier = np.all(np.abs(pc_vis-med) < 5.0*mad, axis=0)
    if inlier.sum() > 50: pc_vis = pc_vis[:, inlier]
    sub    = np.random.choice(pc_vis.shape[1], max(1, pc_vis.shape[1]//100), replace=False)
    pc_sub = pc_vis[:, sub]

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=pc_sub[0], y=pc_sub[1], z=pc_sub[2],
                               mode="markers",
                               marker=dict(size=1.5, color=pc_sub[2],
                                           colorscale="Viridis", opacity=0.5),
                               name="Point Cloud"))
    for i, box in enumerate(predicted):
        add_box(fig, box, "red",   f"Pred {i+1}")
    gt_path = os.path.join(sample_dir, "bbox3d.npy")
    if os.path.exists(gt_path):
        for i, box in enumerate(np.load(gt_path, allow_pickle=True)):
            add_box(fig, box, "green", f"GT {i+1}")

    fig.update_layout(title=f"3D Boxes: {folder}",
                      scene=dict(aspectmode="data"),
                      margin=dict(l=0,r=0,b=0,t=40))
    out = os.path.join(out_dir, f"{folder}_interactive.html")
    fig.write_html(out)
    print(f"  Saved: {out}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = DGCNNBBox(in_channels=ModelConfig.in_channels).to(device)
    if os.path.exists(args.weights):
        ckpt  = torch.load(args.weights, map_location=device)
        state = {k.replace("_orig_mod.",""):v for k,v in ckpt.get("model",ckpt).items()}
        model.load_state_dict(state, strict=False)
        print(f"Loaded: epoch={ckpt.get('epoch','?')} best_mcd={ckpt.get('best_mcd',float('nan')):.4f}m")
    model.eval()
    os.makedirs(args.out_dir, exist_ok=True)

    samples = [args.sample] if args.sample else sorted(
        p for p in (os.path.join(args.data_root,d) for d in os.listdir(args.data_root))
        if os.path.isdir(p) and os.path.exists(os.path.join(p,"rgb.jpg")))

    for i, sd in enumerate(samples, 1):
        run_sample(model, device, sd, args.out_dir, i, len(samples),
                   DataConfig.num_points)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights",  default=DEFAULT_CHECKPOINT)
    p.add_argument("--out_dir",  default=DEFAULT_OUTPUT_DIR)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--sample",    type=str)
    g.add_argument("--data_root", type=str)
    main(p.parse_args())