#!/usr/bin/env python3
"""Utility to create train/val splits for KITTI (Chen split)."""

import os
import argparse
import random


def create_kitti_splits(data_root, seed=42):
    """Create train/val ImageSets following the standard Chen split.

    If ImageSets files already exist, skip.
    Otherwise, split 7481 training samples into ~3712 train + 3769 val.
    """
    imageset_dir = os.path.join(data_root, "ImageSets")
    os.makedirs(imageset_dir, exist_ok=True)

    train_file = os.path.join(imageset_dir, "train.txt")
    val_file   = os.path.join(imageset_dir, "val.txt")

    if os.path.exists(train_file) and os.path.exists(val_file):
        print("ImageSets already exist, skipping.")
        return

    # Find all image IDs
    img_dir = os.path.join(data_root, "training", "image_2")
    if not os.path.isdir(img_dir):
        print(f"ERROR: {img_dir} not found. Download KITTI data first.")
        return

    ids = sorted([
        os.path.splitext(f)[0]
        for f in os.listdir(img_dir)
        if f.endswith(".png")
    ])
    print(f"Found {len(ids)} images.")

    random.seed(seed)
    random.shuffle(ids)
    split_idx = len(ids) // 2

    train_ids = sorted(ids[:split_idx])
    val_ids   = sorted(ids[split_idx:])

    with open(train_file, "w") as f:
        f.write("\n".join(train_ids) + "\n")
    with open(val_file, "w") as f:
        f.write("\n".join(val_ids) + "\n")

    print(f"Created {train_file} ({len(train_ids)} samples)")
    print(f"Created {val_file}   ({len(val_ids)} samples)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="./data/kitti")
    args = p.parse_args()
    create_kitti_splits(args.data_root)