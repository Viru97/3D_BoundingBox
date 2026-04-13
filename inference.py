"""
inference.py  —  (v3)
======================
Changes:
  • Interactive Plotly 3D Bounding Box visualisation output (.html).
  • Fixed tuple unpacking to match the T-Net free 6D PointNet.
  • Prevents GUI hanging on remote headless Linux systems.
"""

import os
import cv2
import torch
import argparse
import numpy as np
import plotly.graph_objects as go
from model import PointNetBBox

EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]


def add_plotly_box(fig, corners, color, name):
    """Draws 3D bounding box edges into a plotly figure."""
    x_lines, y_lines, z_lines = [], [], []
    for i, j in EDGES:
        x_lines.extend([corners[i, 0], corners[j, 0], None])
        y_lines.extend([corners[i, 1], corners[j, 1], None])
        z_lines.extend([corners[i, 2], corners[j, 2], None])

    fig.add_trace(go.Scatter3d(
        x=x_lines, y=y_lines, z=z_lines,
        mode='lines',
        line=dict(color=color, width=4),
        name=name
    ))


def run_sample(model, device, sample_dir, out_dir, sample_idx, total, num_points=1024):
    img_bgr = cv2.imread(os.path.join(sample_dir, "rgb.jpg"))
    if img_bgr is None:
        print(f"  [SKIP] No rgb.jpg in {sample_dir}");
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pc = np.load(os.path.join(sample_dir, "pc.npy"), allow_pickle=True).astype(np.float32)
    masks = np.load(os.path.join(sample_dir, "mask.npy"), allow_pickle=True)

    folder_name = os.path.basename(sample_dir.rstrip("/"))
    print(f"\n[{sample_idx}/{total}] {folder_name} — {len(masks)} object(s)")

    predicted_boxes = []

    for i, mask in enumerate(masks):
        y_idx, x_idx = np.where(mask)
        if len(y_idx) == 0:
            continue

        pc_pts = pc[:, y_idx, x_idx]
        rgb_pts = img_rgb[y_idx, x_idx].astype(np.float32).T / 255.0

        # Step 1: remove zero/missing depth points
        valid = pc_pts[2] > 0.01
        if valid.sum() > 10:
            pc_pts, rgb_pts = pc_pts[:, valid], rgb_pts[:, valid]

        # Step 2: robust outlier removal
        if pc_pts.shape[1] > 10:
            med = np.median(pc_pts, axis=1, keepdims=True)
            mad = np.median(np.abs(pc_pts - med), axis=1, keepdims=True) + 1e-6
            inlier = np.all(np.abs(pc_pts - med) < 5.0 * mad, axis=0)
            if inlier.sum() > 10:
                pc_pts, rgb_pts = pc_pts[:, inlier], rgb_pts[:, inlier]

        # Median anchor
        anchor = np.median(pc_pts, axis=1)
        P = pc_pts.shape[1]

        # Deterministic sampling at inference
        rng = np.random.default_rng(seed=i)
        choice = rng.choice(P, num_points, replace=(P < num_points))
        pc_pts, rgb_pts = pc_pts[:, choice], rgb_pts[:, choice]

        pc_c = pc_pts - anchor.reshape(3, 1)
        features = np.concatenate([pc_c, rgb_pts], axis=0)
        inp = torch.from_numpy(features).unsqueeze(0).to(device)

        with torch.no_grad():
            # FIX: Unpack exactly 3 values from active PointNet model
            center, log_dims, rot6d = model(inp)
            pred_corners = model.get_3d_box(center, log_dims, rot6d)

        corners_abs = pred_corners[0].cpu().numpy() + anchor
        predicted_boxes.append(corners_abs)

        dims = torch.exp(log_dims[0]).cpu().numpy()
        print(f"  Obj {i + 1}: {dims[0]:.3f}m × {dims[1]:.3f}m × {dims[2]:.3f}m")

    # ── Interactive Plotly Figure ──────────────────────────────────────────────
    fig = go.Figure()

    # Add Point Cloud
    pc_flat = pc.reshape(3, -1)
    valid = pc_flat[2] > 0.01
    pc_vis = pc_flat[:, valid].copy()

    if pc_vis.shape[1] > 100:
        med = np.median(pc_vis, axis=1, keepdims=True)
        mad = np.median(np.abs(pc_vis - med), axis=1, keepdims=True) + 1e-6
        inlier = np.all(np.abs(pc_vis - med) < 5.0 * mad, axis=0)
        if inlier.sum() > 50: pc_vis = pc_vis[:, inlier]

    n_sub = max(1, pc_vis.shape[1] // 100)
    sub = np.random.choice(pc_vis.shape[1], n_sub, replace=False)
    pc_sub = pc_vis[:, sub]

    fig.add_trace(go.Scatter3d(
        x=pc_sub[0], y=pc_sub[1], z=pc_sub[2],
        mode='markers',
        marker=dict(size=1.5, color=pc_sub[2], colorscale='Viridis', opacity=0.5),
        name='Point Cloud'
    ))

    # Add Predicted Boxes
    for i, box in enumerate(predicted_boxes):
        add_plotly_box(fig, box, 'red', f'Pred Obj {i + 1}')

    # Add Ground Truth Boxes
    gt_path = os.path.join(sample_dir, "bbox3d.npy")
    if os.path.exists(gt_path):
        gt_boxes = np.load(gt_path, allow_pickle=True)
        for i, box in enumerate(gt_boxes):
            add_plotly_box(fig, box, 'green', f'GT Obj {i + 1}')

    # Configure layout for a clean 3D scene
    fig.update_layout(
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
            aspectmode='data'
        ),
        title=f"Interactive 3D Bounding Boxes: {folder_name}",
        margin=dict(l=0, r=0, b=0, t=40)
    )

    out_path = os.path.join(out_dir, f"{folder_name}_interactive.html")
    fig.write_html(out_path)
    print(f"  Saved Interactive 3D Plot: {out_path}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = PointNetBBox(in_channels=6, num_points=args.num_points).to(device)

    if os.path.exists(args.weights):
        ckpt = torch.load(args.weights, map_location=device)
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state)
        epoch = ckpt.get("epoch", "?")
        mcd = ckpt.get("best_mcd", "?")
        print(f"Loaded weights (epoch={epoch}, best_val_mcd={mcd})")
    else:
        print(f"[WARN] Weights not found at {args.weights} — using random init")

    model.eval()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.data_root:
        samples = sorted(
            p for p in (os.path.join(args.data_root, d) for d in os.listdir(args.data_root))
            if os.path.isdir(p) and os.path.exists(os.path.join(p, "rgb.jpg")))
    else:
        samples = [s for s in args.samples if os.path.isdir(s)]

    if not samples:
        print("No valid sample directories found.");
        return

    print(f"Running inference on {len(samples)} sample(s) …")
    for idx, sd in enumerate(samples, 1):
        run_sample(model, device, sd, args.out_dir, idx, len(samples), num_points=args.num_points)
    print(f"\nDone. Output saved to: {args.out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="best_model.pth")
    p.add_argument("--out_dir", default="output")
    p.add_argument("--num_points", type=int, default=1024)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--samples", nargs="+")
    g.add_argument("--data_root", type=str)
    main(p.parse_args())