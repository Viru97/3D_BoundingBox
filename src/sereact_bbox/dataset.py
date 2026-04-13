"""
dataset.py — PointCloudInstanceDataset
=======================================
7-channel contextual sampling: 512 object points + 512 background points.
The binary mask channel (7th) tells the network which points are the target.

Outlier removal uses MAD (Median Absolute Deviation, 50% breakdown point)
instead of a fixed depth range or IQR (25% breakdown point). This handles
cases where background pixels leak into the instance mask and sit at very
different depths (e.g. Z=2.8m while the object is at Z=1.1m).
"""

import os
import glob
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


def _mad_filter(pc_pts, rgb_pts, threshold=5.0):
    """
    Remove outlier points whose XYZ deviates more than threshold*MAD
    from the median in any axis.

    MAD breakdown point = 50%: survives even if half the points are garbage.
    IQR breakdown point = 25%: fails when >25% of mask pixels are background.
    """
    if pc_pts.shape[1] < 10:
        return pc_pts, rgb_pts
    med = np.median(pc_pts, axis=1, keepdims=True)             # (3,1)
    mad = np.median(np.abs(pc_pts - med), axis=1, keepdims=True) + 1e-6  # (3,1)
    inlier = np.all(np.abs(pc_pts - med) < threshold * mad, axis=0)
    if inlier.sum() < 10:
        return pc_pts, rgb_pts
    return pc_pts[:, inlier], rgb_pts[:, inlier]


class PointCloudInstanceDataset(Dataset):
    def __init__(self, data_root, num_points=1024, is_train=True):
        super().__init__()
        self.num_points = num_points
        self.is_train   = is_train

        self.instances = []
        folders = sorted(f for f in glob.glob(os.path.join(data_root, "*"))
                         if os.path.isdir(f))
        print("Scanning dataset for contextual objects...")
        for folder in folders:
            bbox_path = os.path.join(folder, "bbox3d.npy")
            if os.path.exists(bbox_path):
                bboxes = np.load(bbox_path, allow_pickle=True)
                for i in range(len(bboxes) if bboxes.ndim >= 2 else 0):
                    self.instances.append((folder, i))

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        folder, inst_idx = self.instances[idx]

        img    = cv2.imread(os.path.join(folder, "rgb.jpg"))
        pc     = np.load(os.path.join(folder, "pc.npy"),
                         allow_pickle=True).astype(np.float32)    # (3,H,W)
        masks  = np.load(os.path.join(folder, "mask.npy"),
                         allow_pickle=True)                        # (N,H,W)
        bboxes = np.load(os.path.join(folder, "bbox3d.npy"),
                         allow_pickle=True).astype(np.float32)    # (N,8,3)

        m = masks[inst_idx]
        obj_y, obj_x = np.where(m)
        bg_y,  bg_x  = np.where(~m)

        if len(obj_y) == 0:
            return {"points": torch.zeros(7, self.num_points),
                    "target": torch.zeros(24)}

        # ── extract points ────────────────────────────────────────────────
        pc_obj  = pc[:, obj_y, obj_x]
        rgb_obj = img[obj_y, obj_x, ::-1].astype(np.float32).T / 255.0  # (3,P)
        pc_bg   = pc[:, bg_y,  bg_x]
        rgb_bg  = img[bg_y,  bg_x,  ::-1].astype(np.float32).T / 255.0

        # ── Step 1: remove missing-depth pixels ───────────────────────────
        valid_obj = pc_obj[2] > 0.01
        pc_obj,  rgb_obj = pc_obj[:, valid_obj],  rgb_obj[:, valid_obj]
        valid_bg  = pc_bg[2] > 0.01
        pc_bg,   rgb_bg  = pc_bg[:, valid_bg],   rgb_bg[:, valid_bg]

        # ── Step 2: MAD outlier removal on object points ──────────────────
        # Removes background pixels that leaked into the instance mask
        # (e.g. wall at Z=2.8m when object is at Z=1.1m)
        pc_obj, rgb_obj = _mad_filter(pc_obj, rgb_obj, threshold=5.0)

        # ── Step 3: median anchor — robust to any remaining outliers ───────
        anchor = np.median(pc_obj, axis=1) if pc_obj.shape[1] > 0 \
                 else np.zeros(3, dtype=np.float32)

        # ── Step 4: contextual sampling (50% obj / 50% background) ────────
        n_obj = self.num_points // 2
        n_bg  = self.num_points - n_obj
        N_obj, N_bg = pc_obj.shape[1], pc_bg.shape[1]

        if N_obj > 0:
            c = np.random.choice(N_obj, n_obj, replace=(N_obj < n_obj))
            pc_obj, rgb_obj = pc_obj[:, c], rgb_obj[:, c]
        else:
            pc_obj  = np.zeros((3, n_obj), dtype=np.float32)
            rgb_obj = np.zeros((3, n_obj), dtype=np.float32)

        if N_bg > 0:
            c = np.random.choice(N_bg, n_bg, replace=(N_bg < n_bg))
            pc_bg, rgb_bg = pc_bg[:, c], rgb_bg[:, c]
        else:
            pc_bg  = np.zeros((3, n_bg), dtype=np.float32)
            rgb_bg = np.zeros((3, n_bg), dtype=np.float32)

        # ── Step 5: assemble 7-channel tensor ─────────────────────────────
        pc_comb   = np.concatenate([pc_obj,  pc_bg],  axis=1) - anchor[:,None]
        rgb_comb  = np.concatenate([rgb_obj, rgb_bg],  axis=1)
        mask_comb = np.concatenate([np.ones((1, n_obj), dtype=np.float32),
                                    np.zeros((1, n_bg),  dtype=np.float32)], axis=1)

        features       = np.concatenate([pc_comb, rgb_comb, mask_comb], axis=0)
        target_offsets = bboxes[inst_idx] - anchor[None]   # (8,3) relative

        # ── Augmentation (training only) ──────────────────────────────────
        if self.is_train:
            # 1. Z-rotation ±180°
            theta = np.random.uniform(0, 2 * np.pi)
            c, s  = np.cos(theta), np.sin(theta)
            R = np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float32)
            features[:3] = R @ features[:3]
            target_offsets = (R @ target_offsets.T).T

            # 2. XY reflections (free 4× augmentation)
            for axis in range(2):
                if np.random.rand() < 0.5:
                    features[axis]          *= -1
                    target_offsets[:, axis] *= -1

            # 3. Scale jitter ±10%
            scale = np.random.uniform(0.9, 1.1)
            features[:3]   *= scale
            target_offsets *= scale

            # 4. Small Z-shift (robustness to depth sensor drift)
            z_shift = np.random.uniform(-0.02, 0.02)
            features[2]           += z_shift
            target_offsets[:, 2]  += z_shift

            # 5. Point dropout (up to 10%)
            if np.random.rand() > 0.5:
                n_drop = np.random.randint(1, self.num_points // 10)
                drop   = np.random.choice(self.num_points, n_drop, replace=False)
                features[:6, drop] = 0.0   # keep mask channel intact

            # 6. XYZ noise
            features[:3] += (np.random.randn(3, self.num_points)
                             .astype(np.float32) * 0.003)

            # 7. RGB colour jitter
            for ci in range(3, 6):
                features[ci] = np.clip(
                    np.random.uniform(0.8, 1.2) * features[ci]
                    + np.random.uniform(-0.05, 0.05), 0.0, 1.0)

        return {
            "points": torch.from_numpy(features.astype(np.float32)),   # (7,N)
            "target": torch.from_numpy(
                target_offsets.flatten().astype(np.float32)),           # (24,)
        }