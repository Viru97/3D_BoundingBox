"""Visualization utilities (unchanged except minor robustness fixes)."""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = [
    (0, 255, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255),
    (255, 128, 0), (128, 255, 0), (0, 128, 255), (255, 0, 128),
    (128, 128, 255), (255, 128, 128), (128, 255, 128), (200, 200, 0),
]


def estimate_intrinsics(pc):
    """Estimate fx, fy, cx, cy from organized PC."""
    X, Y, Z = pc[0], pc[1], pc[2]
    H, W = Z.shape
    valid = Z > 0.01
    if valid.sum() < 100:
        return float(W), float(W), W / 2.0, H / 2.0

    vs, us = np.where(valid)
    if len(vs) > 5000:
        idx = np.random.RandomState(0).choice(len(vs), 5000, replace=False)
        vs, us = vs[idx], us[idx]

    xz = X[vs, us] / Z[vs, us]
    yz = Y[vs, us] / Z[vs, us]
    u = us.astype(np.float64)
    v = vs.astype(np.float64)

    A_x = np.stack([xz, np.ones_like(xz)], axis=1)
    fx, cx = np.linalg.lstsq(A_x, u, rcond=None)[0]
    A_y = np.stack([yz, np.ones_like(yz)], axis=1)
    fy, cy = np.linalg.lstsq(A_y, v, rcond=None)[0]

    return float(fx), float(fy), float(cx), float(cy)


def project_3d(pts, fx, fy, cx, cy):
    z = pts[:, 2].copy()
    z[z < 0.001] = 0.001
    u = fx * pts[:, 0] / z + cx
    v = fy * pts[:, 1] / z + cy
    return np.stack([u, v], axis=1), z > 0.01


def draw_box_3d(img, c2d, color, thickness=2):
    c = c2d.astype(int)
    for i in range(4):
        cv2.line(img, tuple(c[i]), tuple(c[(i + 1) % 4]), color, thickness)
    for i in range(4):
        cv2.line(img, tuple(c[4+i]), tuple(c[4+(i+1)%4]), color, thickness)
    for i in range(4):
        cv2.line(img, tuple(c[i]), tuple(c[i + 4]), color, thickness)


def visualize_detections(img_rgb, dets, pc, score_thresh=0.1, gt_corners=None):
    """Draw 3D boxes on image. Returns BGR."""
    vis = img_rgb.copy()
    fx, fy, cx, cy = estimate_intrinsics(pc)

    if gt_corners is not None:
        for gc in gt_corners:
            pts_2d, valid = project_3d(gc, fx, fy, cx, cy)
            if valid.sum() >= 6:
                draw_box_3d(vis, pts_2d, (255, 100, 100), 1)

    if dets is not None and len(dets.get("scores", [])) > 0:
        order = np.argsort(-dets["scores"])
        for rank, j in enumerate(order):
            if dets["scores"][j] < score_thresh:
                continue
            color = COLORS[rank % len(COLORS)]
            pts_2d, valid = project_3d(
                dets["corners"][j], fx, fy, cx, cy)
            if valid.sum() < 6:
                continue
            draw_box_3d(vis, pts_2d, color, 2)
            top = tuple(pts_2d[4].astype(int))
            cv2.putText(vis, f"{dets['scores'][j]:.2f}", top,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)


def visualize_bev(dets, gt_corners=None,
                  x_range=(-0.6, 0.6), z_range=(0.5, 1.5), scale=500):
    W = int((x_range[1] - x_range[0]) * scale)
    H = int((z_range[1] - z_range[0]) * scale)
    bev = np.ones((H, W, 3), dtype=np.uint8) * 240

    def w2b(x, z):
        return (max(0, min(int((x - x_range[0]) * scale), W - 1)),
                max(0, min(int((z_range[1] - z) * scale), H - 1)))

    if gt_corners is not None:
        for gc in gt_corners:
            pts = np.array([w2b(gc[i, 0], gc[i, 2]) for i in range(4)])
            cv2.polylines(bev, [pts], True, (255, 0, 0), 2)

    if dets and len(dets.get("scores", [])) > 0:
        for j in range(len(dets["scores"])):
            color = COLORS[j % len(COLORS)]
            c = dets["corners"][j]
            pts = np.array([w2b(c[i, 0], c[i, 2]) for i in range(4)])
            cv2.polylines(bev, [pts], True, color, 2)

    return bev


def visualize_depth_map(pc, max_depth=1.5):
    z = pc[2] if pc.ndim == 3 else pc
    valid = z > 0.01
    norm = np.zeros_like(z)
    norm[valid] = z[valid] / max_depth
    colored = cv2.applyColorMap(
        (np.clip(norm, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored


def save_training_curves(csv_path, save_path):
    import csv
    epochs, data = [], {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        keys = None
        for row in reader:
            if keys is None:
                keys = [k for k in row.keys()
                        if k not in ("epoch", "lr")]
            epochs.append(int(row["epoch"]))
            for k in keys:
                try:
                    data.setdefault(k, []).append(float(row[k]))
                except ValueError:
                    data.setdefault(k, []).append(0.0)

    n = len(keys)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
    axes = np.array(axes).flatten() if n > 1 else [axes]
    for ax, k in zip(axes, keys):
        ax.plot(epochs, data[k], linewidth=1.2)
        ax.set_title(k, fontsize=11)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
    for ax in axes[n:]:
        ax.set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()