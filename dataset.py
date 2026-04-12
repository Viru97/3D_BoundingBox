import os
import glob
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


class PointCloudInstanceDataset(Dataset):
    """
    Given that we have masks, this dataset extracts individual objects
    as isolated Point Clouds (N=1024 points) with RGB features.
    """

    def __init__(self, data_root, num_points=1024, is_train=True):
        super().__init__()
        self.data_root = data_root
        self.num_points = num_points
        self.is_train = is_train

        # Pre-scan dataset to find all individual instances
        self.instances = []
        folders = [f for f in sorted(glob.glob(os.path.join(data_root, "*"))) if os.path.isdir(f)]

        print("Scanning dataset for objects...")
        for folder in folders:
            bbox_path = os.path.join(folder, "bbox3d.npy")
            if os.path.exists(bbox_path):
                bboxes = np.load(bbox_path, allow_pickle=True)
                n_objects = len(bboxes) if bboxes.ndim >= 2 else 0
                for inst_idx in range(n_objects):
                    self.instances.append((folder, inst_idx))
        print(self.instances)
        print(f"Found {len(self.instances)} individual objects.")

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        folder, inst_idx = self.instances[idx]

        # 1. Load Data
        img_path = os.path.join(folder, "rgb.jpg")
        pc_path = os.path.join(folder, "pc.npy")
        mask_path = os.path.join(folder, "mask.npy")
        bbox_path = os.path.join(folder, "bbox3d.npy")

        img = cv2.imread(img_path)  # (H, W, 3) BGR
        pc = np.load(pc_path, allow_pickle=True).astype(np.float32)  # (3, H, W)
        masks = np.load(mask_path, allow_pickle=True)  # (N, H, W) boolean
        bboxes = np.load(bbox_path, allow_pickle=True).astype(np.float32)  # (N, 8, 3)

        # 2. Extract specific instance
        mask = masks[inst_idx]
        y_idx, x_idx = np.where(mask)

        if len(y_idx) == 0:
            # Failsafe for empty mask
            pc_points = np.zeros((3, self.num_points), dtype=np.float32)
            rgb_points = np.zeros((3, self.num_points), dtype=np.float32)
            target_corners = np.zeros((24,), dtype=np.float32)
            return {'points': torch.from_numpy(np.concatenate([pc_points, rgb_points], axis=0)),
                    'target': torch.from_numpy(target_corners)}

        # Extract XYZ and RGB
        pc_points = pc[:, y_idx, x_idx]  # (3, P)

        # Convert BGR to RGB and normalize
        rgb_points = img[y_idx, x_idx, ::-1].astype(np.float32) / 255.0  # (P, 3)
        rgb_points = rgb_points.transpose(1, 0)  # (3, P)

        # 3. Filter invalid depth points (z <= 0.01) BEFORE sampling and anchor computation
        valid_depth = (pc_points[2] > 0.1) & (pc_points[2] < 1.5)

        if valid_depth.sum() > 10:
            pc_points = pc_points[:, valid_depth]
            rgb_points = rgb_points[:, valid_depth]

        # 4. Geometry Centering (Crucial for translation invariance)
        # Compute anchor on FULL (unsampled) valid point cloud for a stable center
        anchor = pc_points.mean(axis=1)

        # 5. Standardize Point Count (Subsample or Pad to 1024)
        P = pc_points.shape[1]
        if P >= self.num_points:
            choice = np.random.choice(P, self.num_points, replace=False)
        else:
            choice = np.random.choice(P, self.num_points, replace=True)

        pc_points = pc_points[:, choice]
        rgb_points = rgb_points[:, choice]

        pc_points_centered = pc_points - anchor.reshape(3, 1)

        # Calculate target box offsets relative to anchor
        target_box = bboxes[inst_idx]  # (8, 3)
        target_offsets = target_box - anchor.reshape(1, 3)

        # 6. Build final tensor (6, 1024) -> [X,Y,Z, R,G,B]
        if self.is_train:
            # Data Augmentation: Add tiny jitter to points to prevent overfitting
            pc_points_centered += np.random.randn(*pc_points_centered.shape).astype(np.float32) * 0.002

        features = np.concatenate([pc_points_centered, rgb_points], axis=0)

        return {
            'points': torch.from_numpy(features),  # (6, 1024)
            'target': torch.from_numpy(target_offsets.flatten())  # (24,)
        }