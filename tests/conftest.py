from pathlib import Path

import cv2
import numpy as np


def make_sample(root: Path, scene_id: str, objects: int = 1, empty_mask: bool = False) -> Path:
    folder = root / scene_id
    folder.mkdir(parents=True, exist_ok=True)
    h, w = 16, 16
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    rgb[..., 0] = (xx * 13) % 255
    rgb[..., 1] = (yy * 17) % 255
    rgb[..., 2] = 80
    cv2.imwrite(str(folder / "rgb.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    pc = np.zeros((3, h, w), dtype=np.float32)
    pc[0] = xx.astype(np.float32) * 0.01
    pc[1] = yy.astype(np.float32) * 0.01
    pc[2] = 0.05

    masks = []
    bboxes = []
    for i in range(objects):
        mask = np.zeros((h, w), dtype=bool)
        y0 = 3 + i
        x0 = 4 + i
        if not empty_mask:
            mask[y0 : y0 + 5, x0 : x0 + 4] = True
            pc[2, mask] = 0.18 + i * 0.01
        masks.append(mask)
        x_min, x_max = x0 * 0.01, (x0 + 3) * 0.01
        y_min, y_max = y0 * 0.01, (y0 + 4) * 0.01
        z_min, z_max = 0.05, 0.18 + i * 0.01
        bboxes.append(
            np.array(
                [
                    [x_min, y_min, z_min],
                    [x_max, y_min, z_min],
                    [x_max, y_max, z_min],
                    [x_min, y_max, z_min],
                    [x_min, y_min, z_max],
                    [x_max, y_min, z_max],
                    [x_max, y_max, z_max],
                    [x_min, y_max, z_max],
                ],
                dtype=np.float32,
            )
        )
    np.save(folder / "pc.npy", pc)
    np.save(folder / "mask.npy", np.stack(masks))
    np.save(folder / "bbox3d.npy", np.stack(bboxes))
    return folder


def make_dataset(root: Path, scenes: int = 10) -> Path:
    data_root = root / "dataset"
    for i in range(scenes):
        make_sample(data_root, f"scene_{i:03d}", objects=1)
    return data_root
