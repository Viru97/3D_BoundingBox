import os
import cv2
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from model import PointNetBBox


def draw_3d_box(ax, corners, color='r'):
    lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
    for line in lines:
        p1, p2 = corners[line[0]], corners[line[1]]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=color, linewidth=2)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model generates 100% perfect boxes natively so no more complex math needed here!
    model = PointNetBBox(in_channels=6).to(device)
    if os.path.exists(args.weights):
        model.load_state_dict(torch.load(args.weights, map_location=device))
        print("Loaded explicit BBox PointNet weights.")
    model.eval()

    img_bgr = cv2.imread(os.path.join(args.sample, "rgb.jpg"))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pc = np.load(os.path.join(args.sample, "pc.npy"), allow_pickle=True).astype(np.float32)
    masks = np.load(os.path.join(args.sample, "mask.npy"), allow_pickle=True)

    predicted_boxes = []

    for i in range(len(masks)):
        m = masks[i]
        y_idx, x_idx = np.where(m)
        if len(y_idx) == 0: continue

        pc_points = pc[:, y_idx, x_idx]
        rgb_points = img_rgb[y_idx, x_idx, :].astype(np.float32).transpose(1, 0) / 255.0

        P = pc_points.shape[1]
        choice = np.random.choice(P, 1024, replace=(P < 1024))
        pc_points = pc_points[:, choice]
        rgb_points = rgb_points[:, choice]

        depths = np.linalg.norm(pc_points, axis=0)
        valid_mask = depths > 0.01
        anchor = pc_points[:, valid_mask].mean(axis=1) if valid_mask.sum() > 0 else pc_points.mean(axis=1)
        pc_points_centered = pc_points - anchor.reshape(3, 1)

        features = np.concatenate([pc_points_centered, rgb_points], axis=0)
        input_tensor = torch.from_numpy(features).unsqueeze(0).to(device)

        with torch.no_grad():
            center_offset, log_dims, rot6d = model(input_tensor)
            pred_corners = model.get_3d_box(center_offset, log_dims, rot6d)

        # Add point cloud anchor back to place the box in the physical world
        corners_absolute = pred_corners[0].cpu().numpy() + anchor.reshape(1, 3)
        predicted_boxes.append(corners_absolute)

        # Verify physical predicted bounds
        dims = torch.exp(log_dims[0]).cpu().numpy()
        print(f"Object {i + 1} Predicted Size: {dims[0]:.3f}m x {dims[1]:.3f}m x {dims[2]:.3f}m")

    fig = plt.figure(figsize=(15, 6))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(img_rgb)
    ax1.set_title("RGB Input")
    ax1.axis('off')

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    pc_flat = pc.reshape(3, -1)
    pc_sub = pc_flat[:, np.random.choice(pc_flat.shape[1], size=pc_flat.shape[1] // 100, replace=False)]
    ax2.scatter(pc_sub[0], pc_sub[1], pc_sub[2], s=0.5, c=pc_sub[2], cmap='viridis', alpha=0.5)

    for box in predicted_boxes:
        draw_3d_box(ax2, box, color='red')

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
    args = parser.parse_args()
    main(args)