"""
inference.py  —  (v2)
======================
Changes:
  • Loads checkpoint dict (saved by new train.py) with graceful fallback
    to bare state_dict (old format).
  • model() now returns 4 values (T_feat) — unpacked correctly.
  • Matches updated depth filter thresholds from dataset.py (0.1–1.5 m).
  • Deterministic point sampling at inference (seed per object).
  • Cleaner 2D projection overlay using estimated pinhole camera.
"""

import os
import cv2
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from model import PointNetBBox


EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
         (0,4),(1,5),(2,6),(3,7)]


# ── pinhole camera from organised point cloud ─────────────────────────────────
def estimate_intrinsics(pc_hw3):
    """Fit fx,fy,cx,cy by least squares on valid (Z>0) pixels."""
    H, W = pc_hw3.shape[:2]
    X, Y, Z = pc_hw3[...,0].ravel(), pc_hw3[...,1].ravel(), pc_hw3[...,2].ravel()
    u = np.tile(np.arange(W), H).astype(np.float32)
    v = np.repeat(np.arange(H), W).astype(np.float32)
    valid = Z > 0.05
    if valid.sum() < 50:
        f = W / (2 * np.tan(np.deg2rad(30)))
        return f, f, W/2, H/2
    xz, yz = X[valid]/Z[valid], Y[valid]/Z[valid]
    fx, cx = np.linalg.lstsq(np.stack([xz, np.ones_like(xz)],1), u[valid], rcond=None)[0]
    fy, cy = np.linalg.lstsq(np.stack([yz, np.ones_like(yz)],1), v[valid], rcond=None)[0]
    return float(fx), float(fy), float(cx), float(cy)


def project(corners_3d, fx, fy, cx, cy):
    X, Y, Z = corners_3d[:,0], corners_3d[:,1], corners_3d[:,2].clip(0.01)
    return np.stack([cx + fx*X/Z, cy + fy*Y/Z], axis=1)


def draw_box_2d(img, corners_2d, color, thickness=2):
    for i, j in EDGES:
        p1 = tuple(corners_2d[i].astype(int))
        p2 = tuple(corners_2d[j].astype(int))
        cv2.line(img, p1, p2, color, thickness)


def draw_3d_box(ax, corners, color, label=None):
    for k, (i, j) in enumerate(EDGES):
        ax.plot([corners[i,0], corners[j,0]],
                [corners[i,1], corners[j,1]],
                [corners[i,2], corners[j,2]],
                color=color, linewidth=2,
                label=(label if k == 0 else None))


# ── per-sample inference ──────────────────────────────────────────────────────
def run_sample(model, device, sample_dir, out_dir, sample_idx, total,
               num_points=1024):
    img_bgr = cv2.imread(os.path.join(sample_dir, "rgb.jpg"))
    if img_bgr is None:
        print(f"  [SKIP] No rgb.jpg in {sample_dir}"); return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pc      = np.load(os.path.join(sample_dir, "pc.npy"),
                      allow_pickle=True).astype(np.float32)   # (3,H,W)
    masks   = np.load(os.path.join(sample_dir, "mask.npy"), allow_pickle=True)
    H, W    = img_rgb.shape[:2]
    pc_hw3  = pc.transpose(1,2,0)                              # (H,W,3)

    folder_name = os.path.basename(sample_dir.rstrip("/"))
    print(f"\n[{sample_idx}/{total}] {folder_name} — {len(masks)} object(s)")

    fx, fy, cx, cy = estimate_intrinsics(pc_hw3)

    predicted_boxes = []
    overlay = img_bgr.copy()

    for i, mask in enumerate(masks):
        y_idx, x_idx = np.where(mask)
        if len(y_idx) == 0:
            continue

        pc_pts  = pc[:, y_idx, x_idx]
        rgb_pts = img_rgb[y_idx, x_idx].astype(np.float32).T / 255.0  # (3,P)

        valid = (pc_pts[2] > 0.1) & (pc_pts[2] < 1.5)
        if valid.sum() > 10:
            pc_pts  = pc_pts[:, valid]
            rgb_pts = rgb_pts[:, valid]

        anchor = pc_pts.mean(axis=1)
        P      = pc_pts.shape[1]

        # Deterministic sampling at inference
        rng    = np.random.default_rng(seed=i)
        choice = rng.choice(P, num_points, replace=(P < num_points))
        pc_pts  = pc_pts[:, choice]
        rgb_pts = rgb_pts[:, choice]

        pc_c     = pc_pts - anchor.reshape(3,1)
        features = np.concatenate([pc_c, rgb_pts], axis=0)
        inp      = torch.from_numpy(features).unsqueeze(0).to(device)

        with torch.no_grad():
            center, log_dims, rot6d, _ = model(inp)   # ← unpack 4 values
            pred_corners = model.get_3d_box(center, log_dims, rot6d)

        corners_abs = pred_corners[0].cpu().numpy() + anchor          # (8,3)
        predicted_boxes.append(corners_abs)

        dims = torch.exp(log_dims[0]).cpu().numpy()
        print(f"  Obj {i+1}: {dims[0]:.3f}m × {dims[1]:.3f}m × {dims[2]:.3f}m")

        # 2D overlay
        c2d = project(corners_abs, fx, fy, cx, cy).astype(int)
        draw_box_2d(overlay, c2d, color=(0,0,255), thickness=2)

    # Draw GT on overlay
    gt_path = os.path.join(sample_dir, "bbox3d.npy")
    gt_boxes = []
    if os.path.exists(gt_path):
        gt_boxes = np.load(gt_path, allow_pickle=True)
        for box in gt_boxes:
            c2d = project(box, fx, fy, cx, cy).astype(int)
            draw_box_2d(overlay, c2d, color=(0,255,0), thickness=2)

    # ── figure ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 6))
    fig.suptitle(f"{folder_name}  [{sample_idx}/{total}]", fontsize=12)

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(img_rgb); ax1.set_title("RGB Input"); ax1.axis("off")

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    ax2.set_title("2D Projection (Green=GT, Red=Pred)"); ax2.axis("off")

    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    pc_flat = pc.reshape(3,-1)
    sub = np.random.choice(pc_flat.shape[1],
                           max(1, pc_flat.shape[1]//100), replace=False)
    ax3.scatter(pc_flat[0,sub], pc_flat[1,sub], pc_flat[2,sub],
                s=0.5, c=pc_flat[2,sub], cmap="viridis", alpha=0.4)
    for box in predicted_boxes:
        draw_3d_box(ax3, box, "red",   label="Predicted")
    for box in gt_boxes:
        draw_3d_box(ax3, box, "green", label="GT")

    handles, labels = ax3.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        seen.setdefault(l, h)
    ax3.legend(seen.values(), seen.keys(), loc="upper left", fontsize=9)
    ax3.set_xlabel("X"); ax3.set_ylabel("Y"); ax3.set_zlabel("Z")
    ax3.set_title("3D Bounding Boxes")

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{folder_name}_inference.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.show()
    plt.close(fig)


# ── entry point ───────────────────────────────────────────────────────────────
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = PointNetBBox(in_channels=6, num_points=args.num_points).to(device)

    if os.path.exists(args.weights):
        ckpt = torch.load(args.weights, map_location=device)
        # Support both new dict format and legacy bare state_dict
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state)
        epoch = ckpt.get("epoch", "?")
        mcd   = ckpt.get("best_mcd", "?")
        print(f"Loaded weights (epoch={epoch}, best_val_mcd={mcd})")
    else:
        print(f"[WARN] Weights not found at {args.weights} — using random init")

    model.eval()
    os.makedirs(args.out_dir, exist_ok=True)

    # Collect samples
    if args.data_root:
        samples = sorted(
            p for p in (os.path.join(args.data_root, d)
                        for d in os.listdir(args.data_root))
            if os.path.isdir(p) and os.path.exists(os.path.join(p, "rgb.jpg")))
    else:
        samples = [s for s in args.samples if os.path.isdir(s)]

    if not samples:
        print("No valid sample directories found."); return

    print(f"Running inference on {len(samples)} sample(s) …")
    for idx, sd in enumerate(samples, 1):
        run_sample(model, device, sd, args.out_dir, idx, len(samples),
                   num_points=args.num_points)
    print(f"\nDone. Output saved to: {args.out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights",     default="best_model.pth")
    p.add_argument("--out_dir",     default="output")
    p.add_argument("--num_points",  type=int, default=1024)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--samples",   nargs="+")
    g.add_argument("--data_root", type=str)
    main(p.parse_args())