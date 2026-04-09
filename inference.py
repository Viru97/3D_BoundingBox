import os
import cv2
import torch
import argparse
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from model import RGBDDetector3D


def rigidify_box(corners):
    """
    Takes 8 independently regressed corners and forces them into a perfect
    rigid orthogonal 3D bounding box (cuboid) using vector projection.
    """
    # 1. Find the central anchor of the messy points
    center = corners.mean(axis=0)

    # 2. Approximate the 3 local axes from the noisy corners
    # X-axis direction
    axis1 = (corners[1] + corners[2] + corners[5] + corners[6]) - \
            (corners[0] + corners[3] + corners[4] + corners[7])
    # Y-axis direction
    axis2 = (corners[3] + corners[2] + corners[7] + corners[6]) - \
            (corners[0] + corners[1] + corners[4] + corners[5])

    # 3. Gram-Schmidt orthogonalization (Force exactly 90 degree angles)
    norm1 = np.linalg.norm(axis1)
    a1 = axis1 / norm1 if norm1 > 1e-6 else np.array([1.0, 0.0, 0.0])

    a2 = axis2 - np.dot(axis2, a1) * a1
    norm2 = np.linalg.norm(a2)
    a2 = a2 / norm2 if norm2 > 1e-6 else np.array([0.0, 1.0, 0.0])

    # Z-axis is the cross product of X and Y (guaranteed orthogonal)
    a3 = np.cross(a1, a2)

    # 4. Project corners onto axes to find average half-extents (width, height, depth)
    w = np.abs(np.dot(corners - center, a1)).mean()
    h = np.abs(np.dot(corners - center, a2)).mean()
    d = np.abs(np.dot(corners - center, a3)).mean()

    # 5. Reconstruct the 8 perfect corners of the cuboid
    new_corners = np.array([
        center - w * a1 - h * a2 - d * a3,  # 0
        center + w * a1 - h * a2 - d * a3,  # 1
        center + w * a1 + h * a2 - d * a3,  # 2
        center - w * a1 + h * a2 - d * a3,  # 3
        center - w * a1 - h * a2 + d * a3,  # 4
        center + w * a1 - h * a2 + d * a3,  # 5
        center + w * a1 + h * a2 + d * a3,  # 6
        center - w * a1 + h * a2 + d * a3  # 7
    ])
    return new_corners


def extract_predictions(heatmap, corner_map, pc_input, conf_thresh=0.15, top_k=15):
    hmax = F.max_pool2d(heatmap, kernel_size=3, padding=1, stride=1)
    keep = (hmax == heatmap).float()
    heatmap = heatmap * keep

    heatmap = heatmap[0, 0].cpu().numpy()
    corner_map = corner_map[0].cpu().numpy()
    pc_input = pc_input[0].cpu()  # (3, 512, 512)

    ys, xs = np.where(heatmap > conf_thresh)
    scores = heatmap[ys, xs]

    predictions = []
    for i in range(len(ys)):
        x, y = xs[i], ys[i]

        # 1. The network predicts relative OFFSETS (shape & size of the box)
        offset_3d = corner_map[:, y, x].reshape(8, 3)

        # 2. Grab the physical anchor point from the input Point Cloud
        in_x, in_y = int(x * 4), int(y * 4)

        # Use a 5x5 patch around the center to find the stable 3D coordinate
        pc_patch = pc_input[:, max(0, in_y - 2):min(512, in_y + 3), max(0, in_x - 2):min(512, in_x + 3)]
        patch_flat = pc_patch.reshape(3, -1)

        # Just safely grab the exact mean of the point cloud surface patch.
        center_3d = patch_flat.mean(dim=1).numpy()

        # 3. Add offsets back to physical anchor to get final bounding box
        corners_3d = offset_3d + center_3d.reshape(1, 3)

        # 4. Force the messy points into a rigid, perfect cuboid
        corners_3d = rigidify_box(corners_3d)

        predictions.append({
            'score': float(scores[i]),
            'corners_3d': corners_3d
        })

    predictions = sorted(predictions, key=lambda k: k['score'], reverse=True)
    return predictions[:top_k]


def draw_3d_box(ax, corners, color='r'):
    # Standard 8-corner connections
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]
    ]
    for line in lines:
        p1, p2 = corners[line[0]], corners[line[1]]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=color, linewidth=2)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RGBDDetector3D().to(device)
    if os.path.exists(args.weights):
        model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    img_bgr = cv2.imread(os.path.join(args.sample, "rgb.jpg"))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    pc = np.load(os.path.join(args.sample, "pc.npy"), allow_pickle=True).astype(np.float32)

    input_size = (512, 512)
    img_res = cv2.resize(img_rgb, input_size)
    pc_res = cv2.resize(pc.transpose(1, 2, 0), input_size, interpolation=cv2.INTER_NEAREST).transpose(2, 0, 1)

    img_t = torch.from_numpy(img_res).permute(2, 0, 1)
    pc_t = torch.from_numpy(pc_res)
    inputs = torch.cat([img_t, pc_t], dim=0).unsqueeze(0).to(device)

    with torch.no_grad():
        hm_pred, corner_pred = model(inputs)

    # We now pass the Point Cloud input directly into extract_predictions
    predictions = extract_predictions(hm_pred, corner_pred, pc_input=pc_t.unsqueeze(0), conf_thresh=args.conf_thresh)

    fig = plt.figure(figsize=(15, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    ax1.set_title("RGB Input")
    ax1.axis('off')

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    pc_flat = pc.reshape(3, -1)
    pc_sub = pc_flat[:, np.random.choice(pc_flat.shape[1], size=pc_flat.shape[1] // 100, replace=False)]
    ax2.scatter(pc_sub[0], pc_sub[1], pc_sub[2], s=0.5, c=pc_sub[2], cmap='viridis', alpha=0.5)

    for idx, pred in enumerate(predictions):
        draw_3d_box(ax2, pred['corners_3d'], color='red')

    if os.path.exists(os.path.join(args.sample, "bbox3d.npy")):
        gt_boxes = np.load(os.path.join(args.sample, "bbox3d.npy"), allow_pickle=True)
        for box in gt_boxes:
            draw_3d_box(ax2, box, color='green')

    plt.tight_layout()
    plt.savefig("inference_output.png", dpi=150)
    print("Saved 'inference_output.png'.")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="best_model.pth")
    parser.add_argument("--sample", type=str, required=True)
    parser.add_argument("--conf_thresh", type=float, default=0.15)
    args = parser.parse_args()
    main(args)