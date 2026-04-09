"""
Dataset loader — fixed version.

Changes from v1:
  - Store GT edge vectors (9 values) in addition to corner offsets (24 values)
  - Better handling of zero-depth PC pixels
  - More robust mask centroid computation

"""

import os
import glob
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A

from config import Config


# ======================================================================

# Gaussian heatmap utilities

# ======================================================================

def gaussian_radius(det_size, min_overlap=0.7):
    h, w = det_size
    a1, b1, c1 = 1, (h + w), w * h * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(max(0, b1 ** 2 - 4 * a1 * c1))
    r1 = (b1 + sq1) / 2
    a2, b2, c2 = 4, 2 * (h + w), (1 - min_overlap) * w * h
    sq2 = np.sqrt(max(0, b2 ** 2 - 4 * a2 * c2))
    r2 = (b2 + sq2) / 2
    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (h + w)
    c3 = (min_overlap - 1) * w * h
    sq3 = np.sqrt(max(0, b3 ** 2 - 4 * a3 * c3))
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def gaussian2D(shape, sigma=1.0):
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    x, y = int(center[0]), int(center[1])
    H, W = heatmap.shape
    left  = min(x, radius)
    right = min(W - x, radius + 1)
    top   = min(y, radius)
    bot   = min(H - y, radius + 1)
    masked_hm = heatmap[y - top:y + bot, x - left:x + right]
    masked_g  = gaussian[radius - top:radius + bot,
                         radius - left:radius + right]
    if min(masked_g.shape) > 0 and min(masked_hm.shape) > 0:
        np.maximum(masked_hm, masked_g * k, out=masked_hm)
    return heatmap


# ======================================================================

# PC utilities

# ======================================================================

def lookup_pc_robust(pc, y, x, window=7):
    """Look up organized PC at (y, x) with fallback neighborhood search."""
    H, W = pc.shape[1], pc.shape[2]
    y = min(max(int(round(y)), 0), H - 1)
    x = min(max(int(round(x)), 0), W - 1)

    val = pc[:, y, x].astype(np.float32)
    if val[2] > 0.01:
        return val

    # Expanding search window
    for w in range(1, window + 1):
        y0, y1 = max(0, y - w), min(H, y + w + 1)
        x0, x1 = max(0, x - w), min(W, x + w + 1)
        patch = pc[:, y0:y1, x0:x1]
        valid = patch[2] > 0.01
        if valid.any():
            return patch[:, valid].mean(axis=1).astype(np.float32)

    return np.zeros(3, dtype=np.float32)


def corners_to_edges(corners):
    """Extract 3 edge vectors from (8, 3) corners.

    Convention verified from dataset:
        e0 = corners[1] - corners[0]
        e1 = corners[3] - corners[0]
        e2 = corners[4] - corners[0]
    """
    e0 = corners[1] - corners[0]
    e1 = corners[3] - corners[0]
    e2 = corners[4] - corners[0]
    return np.concatenate([e0, e1, e2]).astype(np.float32)  # (9,)


# ======================================================================

# Dataset

# ======================================================================

class DLChallengeDataset(Dataset):
    RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    RGB_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, cfg: Config, sample_ids: list, augment: bool = True):
        self.cfg = cfg
        self.sample_ids = sample_ids
        self.augment = augment

        self.xyz_mean, self.xyz_std = self._compute_xyz_stats()
        print(f"  XYZ stats: mean={self.xyz_mean}, std={self.xyz_std}")

        self.photo_aug = A.Compose([
            A.ColorJitter(brightness=0.3, contrast=0.3,
                          saturation=0.3, hue=0.05, p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        ]) if augment else None

    def _compute_xyz_stats(self):
        all_vals = []
        rng = np.random.RandomState(42)
        for sid in self.sample_ids[:min(50, len(self.sample_ids))]:
            pc = np.load(os.path.join(self.cfg.data_root, sid, "pc.npy"))
            valid = pc[2] > 0.01
            if valid.any():
                xyz = pc[:, valid].astype(np.float64)
                if xyz.shape[1] > 5000:
                    idx = rng.choice(xyz.shape[1], 5000, replace=False)
                    xyz = xyz[:, idx]
                all_vals.append(xyz)

        if all_vals:
            cat = np.concatenate(all_vals, axis=1)
            return (cat.mean(axis=1).astype(np.float32),
                    np.maximum(cat.std(axis=1).astype(np.float32), 0.01))
        return (np.array([0.0, 0.0, 1.0], dtype=np.float32),
                np.array([0.3, 0.3, 0.2], dtype=np.float32))

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sid = self.sample_ids[idx]
        folder = os.path.join(self.cfg.data_root, sid)

        # ---- Load ----
        img = cv2.imread(os.path.join(folder, "rgb.jpg"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]

        pc_raw = np.load(os.path.join(folder, "pc.npy")).astype(np.float32)
        bbox3d = np.load(os.path.join(folder, "bbox3d.npy")).astype(np.float32)
        masks  = np.load(os.path.join(folder, "mask.npy"))

        inp_h, inp_w = self.cfg.input_height, self.cfg.input_width

        # ---- Resize ----
        img = cv2.resize(img, (inp_w, inp_h))

        pc_hwc = pc_raw.transpose(1, 2, 0)
        pc_hwc = cv2.resize(pc_hwc, (inp_w, inp_h),
                            interpolation=cv2.INTER_LINEAR)
        pc = pc_hwc.transpose(2, 0, 1).astype(np.float32)

        N = masks.shape[0]
        masks_r = np.zeros((N, inp_h, inp_w), dtype=bool)
        for i in range(N):
            masks_r[i] = cv2.resize(
                masks[i].astype(np.uint8), (inp_w, inp_h),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)

        # ---- Augmentation ----
        if self.augment:
            img, pc, masks_r, bbox3d = self._augment(img, pc, masks_r, bbox3d)

        if self.photo_aug is not None:
            img = self.photo_aug(image=img)["image"]

        # ---- Normalize RGB ----
        rgb = img.astype(np.float32) / 255.0
        rgb = (rgb - self.RGB_MEAN) / self.RGB_STD
        rgb = np.ascontiguousarray(rgb.transpose(2, 0, 1))

        # ---- Normalize XYZ ----
        valid_depth = pc[2] > 0.01
        xyz = np.zeros_like(pc)
        for c in range(3):
            xyz[c] = (pc[c] - self.xyz_mean[c]) / self.xyz_std[c]
        xyz[:, ~valid_depth] = 0.0

        # ---- Stack ----
        input_tensor = np.concatenate([rgb, xyz], axis=0)  # (6, H, W)

        # ---- Targets ----
        targets = self._build_targets(bbox3d, masks_r, pc)

        meta = {
            "sample_id": sid,
            "pc_raw": pc.copy(),
        }
        return input_tensor, targets, meta

    def _augment(self, img, pc, masks, bbox3d):
        if random.random() < 0.5:
            img   = img[:, ::-1, :].copy()
            pc    = pc[:, :, ::-1].copy()
            pc[0] = -pc[0]
            masks = masks[:, :, ::-1].copy()
            bbox3d = bbox3d.copy()
            bbox3d[:, :, 0] = -bbox3d[:, :, 0]
        return img, pc, masks, bbox3d

    def _build_targets(self, bbox3d, masks, pc):
        nC = self.cfg.num_classes
        oH = self.cfg.output_height
        oW = self.cfg.output_width
        M  = self.cfg.max_objects

        heatmap        = np.zeros((nC, oH, oW), dtype=np.float32)
        offset_2d      = np.zeros((M, 2), dtype=np.float32)
        center_offset  = np.zeros((M, 3), dtype=np.float32)
        gt_edges       = np.zeros((M, 9), dtype=np.float32)
        gt_corners     = np.zeros((M, 24), dtype=np.float32)
        indices        = np.zeros((M,), dtype=np.int64)
        mask_valid     = np.zeros((M,), dtype=np.float32)

        N = min(bbox3d.shape[0], M)
        valid_count = 0

        for i in range(N):
            ys, xs = np.where(masks[i])
            if len(ys) < 10:  # skip tiny / empty masks
                continue

            cx_img = float(np.mean(xs))
            cy_img = float(np.mean(ys))

            fx = cx_img / self.cfg.down_ratio
            fy = cy_img / self.cfg.down_ratio

            if fx < 0 or fx >= oW or fy < 0 or fy >= oH:
                continue

            ix, iy = int(fx), int(fy)

            # Gaussian
            bb_h = (ys.max() - ys.min() + 1) / self.cfg.down_ratio
            bb_w = (xs.max() - xs.min() + 1) / self.cfg.down_ratio
            if bb_h > 0 and bb_w > 0:
                r = max(0, int(gaussian_radius((bb_h, bb_w))))
                r = max(r, 2)  # minimum radius
                draw_gaussian(heatmap[0], (ix, iy), r)

            corners = bbox3d[i]                    # (8, 3)
            center_3d = corners.mean(axis=0)       # (3,)
            pc_ref = lookup_pc_robust(pc, cy_img, cx_img)

            # Validate PC lookup
            if pc_ref[2] < 0.01:
                continue

            j = valid_count
            idx_flat = iy * oW + ix
            indices[j]       = idx_flat
            mask_valid[j]    = 1.0
            offset_2d[j]     = [fx - ix, fy - iy]
            center_offset[j] = center_3d - pc_ref
            EDGE_SCALE = 0.1  # meters
            gt_edges[j] = corners_to_edges(corners) / EDGE_SCALE
            gt_corners[j] = (corners - center_3d).flatten() / EDGE_SCALE
            valid_count += 1

        return {
            "heatmap":       heatmap,
            "offset_2d":     offset_2d,
            "center_offset": center_offset,
            "gt_edges":      gt_edges,
            "gt_corners":    gt_corners,
            "indices":       indices,
            "mask":          mask_valid,
        }


# ======================================================================

# Collate & DataLoader

# ======================================================================

def collate_fn(batch):
    imgs, targets_list, metas_list = zip(*batch)
    imgs = torch.from_numpy(np.stack(imgs))
    targets = {}
    for key in targets_list[0]:
        targets[key] = torch.from_numpy(
            np.stack([t[key] for t in targets_list])
        )
    metas = {
        "sample_id": [m["sample_id"] for m in metas_list],
        "pc_raw": torch.from_numpy(
            np.stack([m["pc_raw"] for m in metas_list])
        ),
    }
    return imgs, targets, metas


def build_splits(cfg: Config):
    all_folders = sorted(glob.glob(os.path.join(cfg.data_root, "*")))
    all_ids = [
        os.path.basename(f) for f in all_folders
        if os.path.isdir(f) and os.path.exists(os.path.join(f, "rgb.jpg"))
    ]
    random.seed(42)
    shuffled = all_ids.copy()
    random.shuffle(shuffled)
    split = int(len(shuffled) * cfg.train_ratio)
    return sorted(shuffled[:split]), sorted(shuffled[split:])


def build_dataloader(cfg, sample_ids, augment, shuffle):
    ds = DLChallengeDataset(cfg, sample_ids, augment=augment)
    return DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=shuffle,
        num_workers=cfg.num_workers, collate_fn=collate_fn,
        pin_memory=True, drop_last=augment,
    )