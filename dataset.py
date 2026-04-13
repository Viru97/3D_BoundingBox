import os
import glob
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

class PointCloudInstanceDataset(Dataset):
    """
    Contextual Sampling Dataset (7-Channels: X, Y, Z, R, G, B, Mask)
    Reverted to lightning-fast Vectorized Random Sampling. DGCNN natively
    handles edge-detection without needing the massive O(N*K) CPU overhead of FPS.
    """
    def __init__(self, data_root, num_points=1024, is_train=True):
        super().__init__()
        self.data_root = data_root
        self.num_points = num_points
        self.is_train = is_train

        self.instances = []
        folders = [f for f in sorted(glob.glob(os.path.join(data_root, "*"))) if os.path.isdir(f)]

        print("Scanning dataset for contextual objects...")
        for folder in folders:
            bbox_path = os.path.join(folder, "bbox3d.npy")
            if os.path.exists(bbox_path):
                bboxes = np.load(bbox_path, allow_pickle=True)
                n_objects = len(bboxes) if bboxes.ndim >= 2 else 0
                for inst_idx in range(n_objects):
                    self.instances.append((folder, inst_idx))

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        folder, inst_idx = self.instances[idx]

        img_path = os.path.join(folder, "rgb.jpg")
        pc_path = os.path.join(folder, "pc.npy")
        mask_path = os.path.join(folder, "mask.npy")
        bbox_path = os.path.join(folder, "bbox3d.npy")

        img = cv2.imread(img_path)
        pc = np.load(pc_path, allow_pickle=True).astype(np.float32)
        masks = np.load(mask_path, allow_pickle=True)
        bboxes = np.load(bbox_path, allow_pickle=True).astype(np.float32)

        m = masks[inst_idx]
        obj_y, obj_x = np.where(m)
        bg_y, bg_x = np.where(~m)

        if len(obj_y) == 0:
            return {'points': torch.zeros((7, self.num_points)), 'target': torch.zeros(24), 'anchor': torch.zeros(3)}

        pc_obj = pc[:, obj_y, obj_x]
        rgb_obj = img[obj_y, obj_x, ::-1].astype(np.float32).transpose(1, 0) / 255.0

        pc_bg = pc[:, bg_y, bg_x]
        rgb_bg = img[bg_y, bg_x, ::-1].astype(np.float32).transpose(1, 0) / 255.0

        valid_obj = np.linalg.norm(pc_obj, axis=0) > 0.01
        pc_obj, rgb_obj = pc_obj[:, valid_obj], rgb_obj[:, valid_obj]

        valid_bg = np.linalg.norm(pc_bg, axis=0) > 0.01
        pc_bg, rgb_bg = pc_bg[:, valid_bg], rgb_bg[:, valid_bg]

        n_obj_samples = self.num_points // 2
        n_bg_samples = self.num_points - n_obj_samples

        N_obj, N_bg = pc_obj.shape[1], pc_bg.shape[1]

        # SPEED OPTIMIZATION: Instant vectorized random choice (Prevents CPU data-starvation)
        if N_obj > 0:
            choice_obj = np.random.choice(N_obj, n_obj_samples, replace=(N_obj < n_obj_samples))
            pc_obj, rgb_obj = pc_obj[:, choice_obj], rgb_obj[:, choice_obj]
        else:
            pc_obj, rgb_obj = np.zeros((3, n_obj_samples), dtype=np.float32), np.zeros((3, n_obj_samples), dtype=np.float32)

        if N_bg > 0:
            choice_bg = np.random.choice(N_bg, n_bg_samples, replace=(N_bg < n_bg_samples))
            pc_bg, rgb_bg = pc_bg[:, choice_bg], rgb_bg[:, choice_bg]
        else:
            pc_bg, rgb_bg = np.zeros((3, n_bg_samples), dtype=np.float32), np.zeros((3, n_bg_samples), dtype=np.float32)

        mask_obj = np.ones((1, n_obj_samples), dtype=np.float32)
        mask_bg = np.zeros((1, n_bg_samples), dtype=np.float32)

        pc_combined = np.concatenate([pc_obj, pc_bg], axis=1)
        rgb_combined = np.concatenate([rgb_obj, rgb_bg], axis=1)
        mask_combined = np.concatenate([mask_obj, mask_bg], axis=1)

        anchor = np.median(pc_obj, axis=1) if N_obj > 0 else np.zeros(3, dtype=np.float32)
        pc_centered = pc_combined - anchor.reshape(3, 1)

        target_box = bboxes[inst_idx]
        target_offsets = target_box - anchor.reshape(1, 3)

        features = np.concatenate([pc_centered, rgb_combined, mask_combined], axis=0)

        if self.is_train:
            theta = np.random.uniform(0, 2*np.pi)
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
            features[:3, :] = R @ features[:3, :]
            target_offsets = (R @ target_offsets.T).T

            for axis in range(2):
                if np.random.rand() < 0.5:
                    features[axis, :]         = -features[axis, :]
                    target_offsets[:, axis]   = -target_offsets[:, axis]

            scale = np.random.uniform(0.9, 1.1)
            features[:3, :] *= scale
            target_offsets *= scale

            z_shift = np.random.uniform(-0.02, 0.02)
            features[2, :] += z_shift
            target_offsets[:, 2] += z_shift

            if np.random.rand() > 0.5:
                drop_n = np.random.randint(1, self.num_points // 10)
                drop_idx = np.random.choice(self.num_points, drop_n, replace=False)
                features[:6, drop_idx] = 0.0

            features[:3, :] += np.random.randn(3, self.num_points).astype(np.float32) * 0.003

            for c_idx in range(3, 6):
                alpha = np.random.uniform(0.8, 1.2)
                beta = np.random.uniform(-0.05, 0.05)
                features[c_idx, :] = np.clip(alpha * features[c_idx, :] + beta, 0.0, 1.0)

        return {
            'points': torch.from_numpy(features.astype(np.float32)),
            'target': torch.from_numpy(target_offsets.flatten().astype(np.float32)),
            'anchor': torch.from_numpy(anchor.astype(np.float32))
        }