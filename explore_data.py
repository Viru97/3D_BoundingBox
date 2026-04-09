#!/usr/bin/env python3
"""
FIRST STEP: Run this to understand the data format.
Usage:  python explore_data.py --data_root ~/Downloads/dl_challenge
"""

import os
import glob
import argparse
import numpy as np
import cv2


def explore_sample(folder):
    """Explore a single sample folder."""
    name = os.path.basename(folder)
    print(f"\n{'='*70}")
    print(f"Sample: {name}")
    print(f"{'='*70}")

    # ---- RGB ----
    img_path = os.path.join(folder, "rgb.jpg")
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        print(f"\n  rgb.jpg:")
        print(f"    shape = {img.shape}  dtype = {img.dtype}")
        print(f"    H×W   = {img.shape[0]}×{img.shape[1]}")
        print(f"    range = [{img.min()}, {img.max()}]")
    else:
        print("  rgb.jpg: NOT FOUND")

    # ---- 3D BBox ----
    bbox_path = os.path.join(folder, "bbox3d.npy")
    if os.path.exists(bbox_path):
        bbox3d = np.load(bbox_path, allow_pickle=True)
        print(f"\n  bbox3d.npy:")
        print(f"    type  = {type(bbox3d)}")
        print(f"    shape = {bbox3d.shape}  dtype = {bbox3d.dtype}")

        if bbox3d.dtype == object:
            # Object array — could be list of dicts/arrays
            print(f"    (object array with {len(bbox3d)} items)")
            for i in range(min(3, len(bbox3d))):
                item = bbox3d[i]
                print(f"    [{i}] type={type(item)}")
                if isinstance(item, dict):
                    print(f"        keys = {list(item.keys())}")
                    for k, v in item.items():
                        v = np.array(v)
                        print(f"        '{k}': shape={v.shape}, "
                              f"dtype={v.dtype}, "
                              f"sample={v.flat[:5]}")
                elif isinstance(item, np.ndarray):
                    print(f"        shape={item.shape} dtype={item.dtype}")
                    print(f"        values={item}")
                elif isinstance(item, (list, tuple)):
                    arr = np.array(item)
                    print(f"        len={len(item)}, as_array_shape={arr.shape}")
                    print(f"        values={arr}")
                else:
                    print(f"        value={item}")
        else:
            # Regular numeric array
            print(f"    ndim  = {bbox3d.ndim}")
            if bbox3d.ndim >= 2:
                print(f"    N_boxes = {bbox3d.shape[0]}")
                print(f"    cols    = {bbox3d.shape[1:]}")
            print(f"    min   = {bbox3d.min(axis=0) if bbox3d.ndim >= 2 else bbox3d.min()}")
            print(f"    max   = {bbox3d.max(axis=0) if bbox3d.ndim >= 2 else bbox3d.max()}")
            if bbox3d.ndim == 2:
                for i in range(min(3, bbox3d.shape[0])):
                    print(f"    row[{i}] = {bbox3d[i]}")
            elif bbox3d.ndim == 3:
                print(f"    box[0] = \n{bbox3d[0]}")
    else:
        print("  bbox3d.npy: NOT FOUND")

    # ---- Point Cloud ----
    pc_path = os.path.join(folder, "pc.npy")
    if os.path.exists(pc_path):
        pc = np.load(pc_path, allow_pickle=True)
        print(f"\n  pc.npy:")
        print(f"    type  = {type(pc)}")
        print(f"    shape = {pc.shape}  dtype = {pc.dtype}")

        if pc.dtype == object:
            print(f"    (object array with {len(pc)} items)")
            for i in range(min(2, len(pc))):
                item = np.array(pc[i])
                print(f"    [{i}] shape={item.shape}")
        else:
            if pc.ndim >= 2:
                print(f"    N_points = {pc.shape[0]}")
                print(f"    channels = {pc.shape[1:]}")
                print(f"    min per col = {pc.min(axis=0)[:min(6, pc.shape[-1])]}")
                print(f"    max per col = {pc.max(axis=0)[:min(6, pc.shape[-1])]}")
                print(f"    mean        = {pc.mean(axis=0)[:min(6, pc.shape[-1])]}")
                print(f"    first 3 rows:\n{pc[:3]}")

                # Check if organized (H, W, 3)# ---- Point Cloud ----
    pc_path = os.path.join(folder, "pc.npy")
    if os.path.exists(pc_path):
        pc = np.load(pc_path, allow_pickle=True)
        print(f"\n  pc.npy:")
        print(f"    type  = {type(pc)}")
        print(f"    shape = {pc.shape}  dtype = {pc.dtype}")

        if pc.dtype == object:
            print(f"    (object array with {len(pc)} items)")
            for i in range(min(2, len(pc))):
                item = np.array(pc[i])
                print(f"    [{i}] shape={item.shape}")
        else:
            if pc.ndim >= 2:
                print(f"    N_points = {pc.shape[0]}")
                if pc.ndim == 3:
                    print(f"    *** Organized point cloud (H={pc.shape[0]}, "
                          f"W={pc.shape[1]}, ch={pc.shape[2]})")
            elif pc.ndim == 1:
                print(f"    1D array, len={len(pc)}")
                print(f"    first 10: {pc[:10]}")
    else:
        print("  pc.npy: NOT FOUND")

    # ---- Instance Mask ----
    mask_path = os.path.join(folder, "mask.npy")
    if os.path.exists(mask_path):
        mask = np.load(mask_path, allow_pickle=True)
        print(f"\n  mask.npy:")
        print(f"    type  = {type(mask)}")
        print(f"    shape = {mask.shape}  dtype = {mask.dtype}")

        if mask.dtype == object:
            print(f"    (object array with {len(mask)} items)")
            for i in range(min(3, len(mask))):
                item = np.array(mask[i])
                print(f"    [{i}] shape={item.shape} dtype={item.dtype}")
        else:
            uniq = np.unique(mask)
            print(f"    unique values ({len(uniq)}): {uniq[:20]}")
            if mask.ndim == 2:
                print(f"    H×W = {mask.shape[0]}×{mask.shape[1]}")
                print(f"    Looks like instance segmentation map "
                      f"({len(uniq)} instances incl. background)")
            elif mask.ndim == 3:
                print(f"    shape = {mask.shape}")
                print(f"    Might be (N_inst, H, W) per-instance masks "
                      f"or (H, W, N_inst)")
    else:
        print("  mask.npy: NOT FOUND")

    return img.shape if os.path.exists(img_path) else None


def check_consistency(data_root):
    """Check consistency across all samples."""
    folders = sorted(glob.glob(os.path.join(data_root, "*")))
    folders = [f for f in folders if os.path.isdir(f)]

    print(f"\n{'#'*70}")
    print(f"# DATASET CONSISTENCY CHECK")
    print(f"# Root: {data_root}")
    print(f"# Total samples: {len(folders)}")
    print(f"{'#'*70}")

    img_shapes = []
    bbox_shapes = []
    pc_shapes = []
    mask_shapes = []
    n_objects_list = []

    for folder in folders:
        img = cv2.imread(os.path.join(folder, "rgb.jpg"))
        if img is not None:
            img_shapes.append(img.shape)

        bbox3d = np.load(os.path.join(folder, "bbox3d.npy"), allow_pickle=True)
        bbox_shapes.append(bbox3d.shape)
        if bbox3d.ndim >= 1:
            n_objects_list.append(bbox3d.shape[0] if bbox3d.ndim >= 2 else len(bbox3d))

        pc = np.load(os.path.join(folder, "pc.npy"), allow_pickle=True)
        pc_shapes.append(pc.shape)

        mask = np.load(os.path.join(folder, "mask.npy"), allow_pickle=True)
        mask_shapes.append(mask.shape)

    # Report
    unique_img = set(img_shapes)
    print(f"\n  Image shapes: {unique_img}")

    unique_bbox_ndim = set(b[1:] if len(b) > 1 else b for b in bbox_shapes)
    print(f"  BBox column dims: {unique_bbox_ndim}")
    print(f"  Objects per image: min={min(n_objects_list)}, "
          f"max={max(n_objects_list)}, "
          f"mean={np.mean(n_objects_list):.1f}")

    unique_pc = set(s[1:] if len(s) > 1 else s for s in pc_shapes)
    print(f"  PC column dims: {unique_pc}")

    unique_mask = set(s for s in mask_shapes)
    print(f"  Mask shapes (unique): {len(unique_mask)} variants")
    if len(unique_mask) <= 5:
        print(f"    {unique_mask}")


def main():
    parser = argparse.ArgumentParser(
        description="Explore the 3D detection challenge dataset"
    )
    parser.add_argument(
        "--data_root", type=str, required=True,
        help="Root directory containing sample folders"
    )
    parser.add_argument(
        "--n_samples", type=int, default=5,
        help="Number of samples to explore in detail"
    )
    args = parser.parse_args()

    folders = sorted(glob.glob(os.path.join(args.data_root, "*")))
    folders = [f for f in folders if os.path.isdir(f)]

    if not folders:
        print(f"No folders found in {args.data_root}")
        return

    print(f"Found {len(folders)} sample folders")

    # Detailed exploration of first N samples
    for folder in folders[:args.n_samples]:
        explore_sample(folder)

    # Consistency check
    check_consistency(args.data_root)


if __name__ == "__main__":
    main()