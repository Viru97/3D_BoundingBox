import os
import cv2
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from model import PointNetBBox


def draw_3d_box(ax, corners, color, label=None):
    lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
    for j, line in enumerate(lines):
        p1, p2 = corners[line[0]], corners[line[1]]
        # Only pass label on the first segment so the legend shows one entry per box type
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                color=color, linewidth=2, label=label if j == 0 else None)


def run_sample(model, device, sample_dir, out_dir, sample_idx, total):
    """Run inference on a single sample directory and save an output image."""
    img_bgr = cv2.imread(os.path.join(sample_dir, "rgb.jpg"))
    if img_bgr is None:
        print(f"  [SKIP] No rgb.jpg in {sample_dir}")
        return
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pc = np.load(os.path.join(sample_dir, "pc.npy"), allow_pickle=True).astype(np.float32)
    masks = np.load(os.path.join(sample_dir, "mask.npy"), allow_pickle=True)

    folder_name = os.path.basename(sample_dir.rstrip("/"))
    print(f"\n[{sample_idx}/{total}] {folder_name} — {len(masks)} object(s)")

    predicted_boxes = []

    for i in range(len(masks)):
        m = masks[i]
        y_idx, x_idx = np.where(m)
        if len(y_idx) == 0:
            continue

        pc_points = pc[:, y_idx, x_idx]
        rgb_points = img_rgb[y_idx, x_idx, :].astype(np.float32).transpose(1, 0) / 255.0

        # Filter invalid depth points (z <= 0.01) — matches training pipeline
        valid_depth = pc_points[2] > 0.01
        if valid_depth.sum() > 10:
            pc_points = pc_points[:, valid_depth]
            rgb_points = rgb_points[:, valid_depth]

        # Compute anchor on full valid point cloud
        anchor = pc_points.mean(axis=1)

        P = pc_points.shape[1]
        choice = np.random.choice(P, 1024, replace=(P < 1024))
        pc_points = pc_points[:, choice]
        rgb_points = rgb_points[:, choice]

        pc_points_centered = pc_points - anchor.reshape(3, 1)

        features = np.concatenate([pc_points_centered, rgb_points], axis=0)
        input_tensor = torch.from_numpy(features).unsqueeze(0).to(device)

        with torch.no_grad():
            center_offset, log_dims, rot6d = model(input_tensor)
            pred_corners = model.get_3d_box(center_offset, log_dims, rot6d)

        corners_absolute = pred_corners[0].cpu().numpy() + anchor.reshape(1, 3)
        predicted_boxes.append(corners_absolute)

        dims = torch.exp(log_dims[0]).cpu().numpy()
        print(f"  Object {i + 1}: {dims[0]:.3f}m x {dims[1]:.3f}m x {dims[2]:.3f}m")

    fig = plt.figure(figsize=(15, 6))
    fig.suptitle(f"{folder_name}  [{sample_idx}/{total}]", fontsize=12)

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(img_rgb)
    ax1.set_title("RGB Input")
    ax1.axis('off')

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    pc_flat = pc.reshape(3, -1)
    pc_sub = pc_flat[:, np.random.choice(pc_flat.shape[1], size=max(1, pc_flat.shape[1] // 100), replace=False)]
    ax2.scatter(pc_sub[0], pc_sub[1], pc_sub[2], s=0.5, c=pc_sub[2], cmap='viridis', alpha=0.5)
    ax2.set_title("3D Bounding Boxes")
    ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z")

    for box in predicted_boxes:
        draw_3d_box(ax2, box, color='red', label='Predicted')

    gt_path = os.path.join(sample_dir, "bbox3d.npy")
    if os.path.exists(gt_path):
        gt_boxes = np.load(gt_path, allow_pickle=True)
        for box in gt_boxes:
            draw_3d_box(ax2, box, color='green', label='Ground Truth')

    # Deduplicate legend entries (one per colour, not one per box)
    handles, labels = ax2.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        seen.setdefault(l, h)
    ax2.legend(seen.values(), seen.keys(), loc='upper left', fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{folder_name}_inference.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {out_path}")

    plt.show()   # Interactive window — close it to continue to the next sample
    plt.close(fig)


def collect_samples(args):
    """Resolve the list of sample directories from CLI args."""
    samples = []

    if args.data_root:
        # Scan an entire directory for all valid sample folders
        for entry in sorted(os.listdir(args.data_root)):
            path = os.path.join(args.data_root, entry)
            if os.path.isdir(path) and os.path.exists(os.path.join(path, "rgb.jpg")):
                samples.append(path)
    else:
        # Explicit list of sample directories
        for s in args.samples:
            if os.path.isdir(s):
                samples.append(s)
            else:
                print(f"[WARN] Not a directory, skipping: {s}")

    return samples


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = PointNetBBox(in_channels=6).to(device)
    if os.path.exists(args.weights):
        model.load_state_dict(torch.load(args.weights, map_location=device))
        print("Loaded weights.")
    model.eval()

    os.makedirs(args.out_dir, exist_ok=True)

    samples = collect_samples(args)
    if not samples:
        print("No valid sample directories found. Use --samples or --data_root.")
        return

    print(f"Running inference on {len(samples)} sample(s)...")
    for idx, sample_dir in enumerate(samples, start=1):
        run_sample(model, device, sample_dir, args.out_dir, idx, len(samples))

    print(f"\nDone. {len(samples)} image(s) saved to: {args.out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="best_model.pth")
    parser.add_argument("--out_dir", type=str, default="output",
                        help="Directory to save output images")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--samples", nargs="+",
                       help="One or more sample directories, e.g. --samples data/001 data/002")
    group.add_argument("--data_root", type=str,
                       help="Root directory — runs on every sub-folder that contains rgb.jpg")

    args = parser.parse_args()
    main(args)