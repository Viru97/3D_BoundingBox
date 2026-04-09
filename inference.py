#!/usr/bin/env python3
"""Run inference on a single image or directory of images."""

import argparse
import os
import cv2
import numpy as np
import torch

from config import Config
from model import CenterNet3D
from decode import decode_predictions
from visualize import visualize_detections, visualize_bev


def preprocess(img_bgr, cfg):
    """Preprocess a single BGR image for inference."""
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (cfg.input_width, cfg.input_height))
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)  # CHW
    return torch.from_numpy(img).float().unsqueeze(0)


def load_calib(calib_path):
    """Load KITTI calibration file."""
    data = {}
    with open(calib_path) as f:
        for line in f:
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            data[k.strip()] = np.array([float(x) for x in v.strip().split()])
    return data["P2"].reshape(3, 4).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image",      type=str, required=True,
                        help="Path to image or directory")
    parser.add_argument("--calib",      type=str, required=True,
                        help="Path to calib file or directory")
    parser.add_argument("--output_dir", type=str, default="./output/inference")
    parser.add_argument("--threshold",  type=float, default=0.3)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg  = ckpt.get("cfg", Config())
    cfg.score_thresh = args.threshold

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CenterNet3D(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    # Gather image paths
    if os.path.isdir(args.image):
        img_paths = sorted([
            os.path.join(args.image, f)
            for f in os.listdir(args.image)
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ])
    else:
        img_paths = [args.image]

    print(f"Running inference on {len(img_paths)} images...")

    for img_path in img_paths:
        name = os.path.splitext(os.path.basename(img_path))[0]

        # Load & preprocess image
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"  Skipping {img_path}")
            continue
        img_tensor = preprocess(img_bgr, cfg).to(device)

        # Load calibration
        if os.path.isdir(args.calib):
            calib_path = os.path.join(args.calib, f"{name}.txt")
        else:
            calib_path = args.calib
        P2 = load_calib(calib_path)

        # Scale calibration
        orig_h, orig_w = img_bgr.shape[:2]
        P2_scaled = P2.copy()
        P2_scaled[0, :] *= cfg.input_width  / orig_w
        P2_scaled[1, :] *= cfg.input_height / orig_h
        calib_t = torch.from_numpy(P2_scaled).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            output = model(img_tensor)
            dets = decode_predictions(output, calib_t, cfg)[0]

        n_det = len(dets["scores"])
        print(f"  {name}: {n_det} detections")

        # Visualize
        img_resized = cv2.resize(img_bgr, (cfg.input_width, cfg.input_height))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        vis = visualize_detections(
            img_rgb, dets, P2_scaled, cfg.classes, score_thresh=args.threshold
        )
        cv2.imwrite(os.path.join(args.output_dir, f"{name}_3d.png"), vis)

        bev = visualize_bev(dets)
        cv2.imwrite(os.path.join(args.output_dir, f"{name}_bev.png"), bev)

        # Print detections
        for i in range(n_det):
            cls_name = cfg.classes[int(dets["cls_ids"][i])]
            sc = dets["scores"][i]
            loc = dets["locs"][i]
            dims = dets["dims"][i]
            print(f"    {cls_name} score={sc:.2f}  "
                  f"xyz=({loc[0]:.1f},{loc[1]:.1f},{loc[2]:.1f})  "
                  f"hwl=({dims[0]:.2f},{dims[1]:.2f},{dims[2]:.2f})")

    print(f"\n✓ Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()