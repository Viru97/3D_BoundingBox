"""
dataset.py  —  PointCloudInstanceDataset  (v2)
================================================
Augmentations added:
  • Random rotation around gravity (Z) axis           ← most impactful
  • Random scale jitter (±10 %)
  • Random axis-aligned reflection (X and Y)
  • Point dropout  (randomly zero-out up to 10 % of points)
  • Colour jitter  (brightness / contrast on the RGB channels)
  • Gaussian point noise (existing, tuned up slightly)

Training-time anchor normalisation is unchanged so the model still
regresses box corners relative to the point-cloud centroid.
"""

import os
import glob
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


# ── rotation helpers ──────────────────────────────────────────────────────────
def _rot_z(angle_rad):
    """3×3 rotation matrix around the Z (gravity) axis."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]], dtype=np.float32)


class PointCloudInstanceDataset(Dataset):
    """
    Extracts individual objects as isolated point clouds (N=1024 points)
    with XYZ + RGB features.  Targets are the 8 GT corners *relative to
    the point-cloud centroid* so the model is translation-invariant.
    """

    def __init__(self, data_root, num_points=1024, is_train=True):
        super().__init__()
        self.data_root  = data_root
        self.num_points = num_points
        self.is_train   = is_train

        self.instances = []
        folders = sorted(f for f in glob.glob(os.path.join(data_root, "*"))
                         if os.path.isdir(f))

        print("Scanning dataset for objects …")
        for folder in folders:
            bbox_path = os.path.join(folder, "bbox3d.npy")
            if os.path.exists(bbox_path):
                bboxes = np.load(bbox_path, allow_pickle=True)
                n = len(bboxes) if bboxes.ndim >= 2 else 0
                for i in range(n):
                    self.instances.append((folder, i))
        print(f"Found {len(self.instances)} individual objects.")

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        folder, inst_idx = self.instances[idx]

        # ── load ──────────────────────────────────────────────────────────
        img   = cv2.imread(os.path.join(folder, "rgb.jpg"))          # (H,W,3) BGR
        pc    = np.load(os.path.join(folder, "pc.npy"),
                        allow_pickle=True).astype(np.float32)         # (3,H,W)
        masks = np.load(os.path.join(folder, "mask.npy"),
                        allow_pickle=True)                             # (N,H,W)
        bboxes = np.load(os.path.join(folder, "bbox3d.npy"),
                         allow_pickle=True).astype(np.float32)        # (N,8,3)

        # ── extract instance ───────────────────────────────────────────────
        mask = masks[inst_idx]
        y_idx, x_idx = np.where(mask)

        if len(y_idx) == 0:
            pts = np.zeros((6, self.num_points), dtype=np.float32)
            tgt = np.zeros((24,), dtype=np.float32)
            return {"points": torch.from_numpy(pts),
                    "target": torch.from_numpy(tgt)}

        # XYZ
        pc_pts = pc[:, y_idx, x_idx]              # (3, P)
        # RGB  (BGR→RGB, normalise)
        rgb_pts = img[y_idx, x_idx, ::-1].astype(np.float32) / 255.0  # (P,3)
        rgb_pts = rgb_pts.T                        # (3, P)

        # ── Step 1: remove zero/invalid points (missing depth) ────────────
        valid = pc_pts[2] > 0.01
        if valid.sum() > 10:
            pc_pts  = pc_pts[:, valid]
            rgb_pts = rgb_pts[:, valid]

        # ── Step 2: robust outlier removal using MAD ──────────────────────
        # IQR breaks when >25% of mask pixels are background leaks.
        # MAD has a 50% breakdown point — survives much heavier contamination.
        if pc_pts.shape[1] > 10:
            med = np.median(pc_pts, axis=1, keepdims=True)       # (3,1)
            mad = np.median(np.abs(pc_pts - med), axis=1,
                            keepdims=True) + 1e-6                 # (3,1)
            inlier = np.all(np.abs(pc_pts - med) < 5.0 * mad, axis=0)
            if inlier.sum() > 10:
                pc_pts  = pc_pts[:, inlier]
                rgb_pts = rgb_pts[:, inlier]

        # ── Step 3: median anchor — robust to remaining outliers ───────────
        anchor = np.median(pc_pts, axis=1)    # (3,)

        # ── subsample / pad ────────────────────────────────────────────────
        P = pc_pts.shape[1]
        choice = np.random.choice(P, self.num_points, replace=(P < self.num_points))
        pc_pts  = pc_pts[:, choice]
        rgb_pts = rgb_pts[:, choice]

        # Centre around anchor
        pc_c = pc_pts - anchor.reshape(3, 1)       # (3, N) centred

        # ── target corners (relative to anchor) ───────────────────────────
        target_box     = bboxes[inst_idx]          # (8, 3) absolute
        target_offsets = target_box - anchor.reshape(1, 3)  # (8, 3) relative

        # ════════════════════════════════════════════════════════════════════
        # AUGMENTATION (training only)
        # ════════════════════════════════════════════════════════════════════
        if self.is_train:

            # 1. Random rotation around Z (gravity) axis — ±180°
            #    Rotation must be applied consistently to points AND box corners.
            angle = np.random.uniform(-np.pi, np.pi)
            R     = _rot_z(angle)                 # (3,3)
            pc_c          = R @ pc_c              # (3,N)
            target_offsets = (R @ target_offsets.T).T  # (8,3)

            # 2. Random scale jitter ±10 %
            scale          = np.random.uniform(0.90, 1.10)
            pc_c          *= scale
            target_offsets *= scale

            # 3. Random axis-aligned reflections (X and/or Y)
            for axis in range(2):
                if np.random.rand() < 0.5:
                    pc_c[axis]           = -pc_c[axis]
                    target_offsets[:, axis] = -target_offsets[:, axis]

            # 4. Point dropout — randomly zero-out up to 10 % of points
            if np.random.rand() < 0.5:
                drop_n = np.random.randint(1, max(2, self.num_points // 10))
                drop_idx = np.random.choice(self.num_points, drop_n, replace=False)
                pc_c[:, drop_idx]  = 0.0
                rgb_pts[:, drop_idx] = 0.0

            # 5. Gaussian noise on XYZ (slightly stronger than before)
            pc_c += np.random.randn(*pc_c.shape).astype(np.float32) * 0.003

            # 6. Colour jitter on RGB channels
            #    Independent brightness/contrast per channel
            for c in range(3):
                alpha = np.random.uniform(0.8, 1.2)   # contrast
                beta  = np.random.uniform(-0.05, 0.05) # brightness
                rgb_pts[c] = np.clip(alpha * rgb_pts[c] + beta, 0.0, 1.0)

        # ── assemble feature tensor (6, N) ────────────────────────────────
        features = np.concatenate([pc_c, rgb_pts], axis=0).astype(np.float32)

        return {
            "points": torch.from_numpy(features),              # (6, 1024)
            "target": torch.from_numpy(target_offsets.flatten().astype(np.float32)),  # (24,)
        }