import os
import glob
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class RGBD3DDataset(Dataset):
    """
    Dataset loader for the 3D Detection Challenge.
    Uses Residual (Offset) Regression for absolute 3D coordinates.
    """

    def __init__(self, data_root, input_size=(512, 512), down_ratio=4, is_train=True):
        super().__init__()
        self.data_root = data_root
        self.input_size = input_size
        self.down_ratio = down_ratio
        self.is_train = is_train

        # Prevent overfitting on small datasets
        self.color_jitter = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
        self.folders = [f for f in sorted(glob.glob(os.path.join(data_root, "*"))) if os.path.isdir(f)]

    def __len__(self):
        return len(self.folders)

    def draw_gaussian(self, heatmap, center, radius=3):
        diameter = 2 * radius + 1
        gaussian = np.zeros((diameter, diameter), dtype=np.float32)
        for i in range(diameter):
            for j in range(diameter):
                dist = np.sqrt((i - radius) ** 2 + (j - radius) ** 2)
                if dist <= radius:
                    gaussian[i, j] = np.exp(-0.5 * (dist / (radius / 2)) ** 2)

        y, x = int(center[1]), int(center[0])
        h, w = heatmap.shape

        left, right = min(x, radius), min(w - x, radius + 1)
        top, bottom = min(y, radius), min(h - y, radius + 1)

        masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]

        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            np.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)

    def __getitem__(self, idx):
        folder = self.folders[idx]

        # 1. Load Data
        img_path = os.path.join(folder, "rgb.jpg")
        pc_path = os.path.join(folder, "pc.npy")
        mask_path = os.path.join(folder, "mask.npy")
        bbox_path = os.path.join(folder, "bbox3d.npy")

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        pc = np.load(pc_path, allow_pickle=True).astype(np.float32)  # (3, H_orig, W_orig)
        masks = np.load(mask_path, allow_pickle=True)  # (N, H_orig, W_orig)
        bboxes = np.load(bbox_path, allow_pickle=True).astype(np.float32)  # (N, 8, 3)

        # 2. Resize Inputs
        orig_h, orig_w = img.shape[:2]
        img_res = cv2.resize(img, self.input_size)

        pc_transposed = pc.transpose(1, 2, 0)
        pc_res = cv2.resize(pc_transposed, self.input_size, interpolation=cv2.INTER_NEAREST)
        pc_res = pc_res.transpose(2, 0, 1)

        # 3. Create Target Maps
        out_h, out_w = self.input_size[0] // self.down_ratio, self.input_size[1] // self.down_ratio

        heatmap = np.zeros((1, out_h, out_w), dtype=np.float32)
        corner_map = np.zeros((24, out_h, out_w), dtype=np.float32)
        reg_mask = np.zeros((1, out_h, out_w), dtype=np.float32)

        if masks.ndim == 3:
            for i in range(len(masks)):
                m = masks[i]
                y_idx, x_idx = np.where(m)
                if len(y_idx) == 0: continue

                # Center for Heatmap placement
                center_y = np.mean(y_idx) * (out_h / orig_h)
                center_x = np.mean(x_idx) * (out_w / orig_w)
                ct_int = [int(center_x), int(center_y)]
                ct_int[0] = max(0, min(out_w - 1, ct_int[0]))
                ct_int[1] = max(0, min(out_h - 1, ct_int[1]))

                self.draw_gaussian(heatmap[0], ct_int, radius=3)

                # --- CRITICAL FIX: RESIDUAL TARGETS ---
                # Calculate the exact physical 3D center of the object using the point cloud mask
                object_pc = pc[:, m]  # (3, N_points_in_mask)
                center_3d = np.mean(object_pc, axis=1)  # (3,) absolute anchor point

                # Subtract the anchor to create completely translation-invariant offsets
                corner_offsets = bboxes[i] - center_3d.reshape(1, 3)  # (8, 3) relative coordinates

                corner_map[:, ct_int[1], ct_int[0]] = corner_offsets.flatten()
                reg_mask[0, ct_int[1], ct_int[0]] = 1.0

        # Convert to Tensors
        img_t = torch.from_numpy(img_res).permute(2, 0, 1)
        pc_t = torch.from_numpy(pc_res)

        if self.is_train:
            img_t = self.color_jitter(img_t)
            pc_t = pc_t + torch.randn_like(pc_t) * 0.002  # noise injection

        inputs = torch.cat([img_t, pc_t], dim=0)

        return {
            'input': inputs,
            'heatmap': torch.from_numpy(heatmap),
            'corner_map': torch.from_numpy(corner_map),
            'reg_mask': torch.from_numpy(reg_mask)
        }