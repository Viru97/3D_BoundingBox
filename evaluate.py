#!/usr/bin/env python3
"""Standalone evaluation with visualizations."""

import argparse
import os
import cv2
import numpy as np
import torch
from tqdm import tqdm

from config import Config

from dataset import DLChallengeDataset, build_splits, collate_fn
from torch.utils.data import DataLoader
from model import CenterNet3D
from decode import decode_predictions
from metrics import compute_all_metrics, iou_3d, corner_distance
from visualize import (visualize_detections, visualize_bev,
                       visualize_depth_map, save_training_curves)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_root",  type=str, default=None)
    parser.add_argument("--vis_dir",    type=str, default="./output/vis")
    parser.add_argument("--max_vis",    type=int, default=40)
    parser.add_argument("--threshold",  type=float, default=0.1)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt.get("cfg", Config())
    if args.data_root:
        cfg.data_root = args.data_root
    cfg.score_thresh = args.threshold
    cfg.batch_size = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CenterNet3D(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    _, val_ids = build_splits(cfg)
    ds = DLChallengeDataset(cfg, val_ids, augment=False)
    if "xyz_mean" in ckpt:
        ds.xyz_mean = ckpt["xyz_mean"]
        ds.xyz_std  = ckpt["xyz_std"]

    loader = DataLoader(ds, batch_size=1, collate_fn=collate_fn, num_workers=2)
    os.makedirs(args.vis_dir, exist_ok=True)

    all_dets, all_gts = [], []

    for i, (imgs, targets, metas) in enumerate(tqdm(loader, desc="Eval")):
        imgs = imgs.to(device)
        pc   = metas["pc_raw"].to(device)

        with torch.no_grad(), torch.amp.autocast('cuda', enabled=cfg.use_amp):
            out = model(imgs)
        dets = decode_predictions(out, pc, cfg)

        det = dets[0]
        all_dets.append(det)

        sid = metas["sample_id"][0]
        bbox3d = np.load(
            os.path.join(cfg.data_root, sid, "bbox3d.npy")).astype(np.float32)
        all_gts.append(bbox3d)

        # ---- Per-image debug ----
        if i < 5:
            print(f"\n  Image {sid[:12]}: "
                  f"{len(det['scores'])} dets, {len(bbox3d)} GT")
            if len(det["scores"]) > 0:
                print(f"    Score range: [{det['scores'].min():.3f}, "
                      f"{det['scores'].max():.3f}]")
                for j_det in range(min(3, len(det["scores"]))):
                    for k_gt in range(min(3, len(bbox3d))):
                        iou = iou_3d(det["corners"][j_det], bbox3d[k_gt])
                        cdist = corner_distance(
                            det["corners"][j_det], bbox3d[k_gt])
                        print(f"    det[{j_det}] vs gt[{k_gt}]: "
                              f"IoU={iou:.4f}, corner_err={cdist:.1f}cm")

        # ---- Visualize ----
        if i < args.max_vis:
            pc_np = metas["pc_raw"][0].numpy()
            img_path = os.path.join(cfg.data_root, sid, "rgb.jpg")
            img_orig = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(
                cv2.resize(img_orig, (cfg.input_width, cfg.input_height)),
                cv2.COLOR_BGR2RGB)

            short = sid[:12]
            vis = visualize_detections(
                img_rgb, det, pc_np, score_thresh=args.threshold,
                gt_corners=bbox3d)
            cv2.imwrite(os.path.join(args.vis_dir, f"{short}_3d.png"), vis)

            bev = visualize_bev(det, gt_corners=bbox3d)
            cv2.imwrite(os.path.join(args.vis_dir, f"{short}_bev.png"), bev)

            dm = visualize_depth_map(pc_np)
            cv2.imwrite(os.path.join(args.vis_dir, f"{short}_depth.png"), dm)

    # ---- Metrics ----
    print(f"\n{'='*65}")
    print("  EVALUATION RESULTS")
    print(f"{'='*65}")
    metrics = compute_all_metrics(all_dets, all_gts, verbose=True)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    csv_path = os.path.join(os.path.dirname(args.checkpoint), "train_log.csv")
    if os.path.exists(csv_path):
        save_training_curves(
            csv_path, os.path.join(args.vis_dir, "training_curves.png"))

    print(f"\n✓ Vis saved to {args.vis_dir}")


if __name__ == "__main__":
    main()