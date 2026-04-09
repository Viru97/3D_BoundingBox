#!/usr/bin/env python3
"""Training entry point."""

import argparse
import os
import random
import numpy as np
import torch

from config import Config
from dataset import build_splits, build_dataloader
from model import CenterNet3D
from engine import Trainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",   type=str, default="./dl_challenge")
    p.add_argument("--output_dir",  type=str, default="./output")
    p.add_argument("--backbone",    type=str, default="resnet18")
    p.add_argument("--epochs",      type=int, default=200)
    p.add_argument("--batch_size",  type=int, default=8)
    p.add_argument("--lr",          type=float, default=5e-4)
    p.add_argument("--input_h",     type=int, default=480)
    p.add_argument("--input_w",     type=int, default=640)
    p.add_argument("--no_amp",      action="store_true")
    p.add_argument("--resume",      type=str, default=None)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    cfg = Config(
        data_root=args.data_root, output_dir=args.output_dir,
        backbone=args.backbone, max_epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr,
        input_height=args.input_h, input_width=args.input_w,
        use_amp=not args.no_amp, num_workers=args.num_workers,
    )
    os.makedirs(cfg.output_dir, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  CenterNet3D — 3D BBox Prediction (v2 - edge vectors)")
    print(f"{'='*65}")
    print(f"  Data       : {cfg.data_root}")
    print(f"  Input      : {cfg.in_channels}ch × {cfg.input_height}×{cfg.input_width}")
    print(f"  Backbone   : {cfg.backbone}")
    print(f"  Batch/Epoch: {cfg.batch_size} / {cfg.max_epochs}")
    print(f"  LR         : {cfg.lr} (warmup {cfg.warmup_epochs}ep)")
    print(f"  AMP        : {cfg.use_amp}")
    print(f"  Heads      : heatmap(1) + offset(2) + center(3) + edges(9)")
    print(f"{'='*65}\n")

    train_ids, val_ids = build_splits(cfg)
    print(f"Split: {len(train_ids)} train, {len(val_ids)} val")

    train_loader = build_dataloader(cfg, train_ids, augment=True, shuffle=True)
    val_loader   = build_dataloader(cfg, val_ids, augment=False, shuffle=False)

    model = CenterNet3D(cfg)
    print(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"Resumed from {args.resume}")

    trainer = Trainer(model, train_loader, val_loader, cfg)
    if args.resume and os.path.isfile(args.resume):
        trainer.optimizer.load_state_dict(ckpt["optim"])
        trainer.scheduler.load_state_dict(ckpt["sched"])
        trainer.best_metric = ckpt.get("best_metric", -1)

    trainer.fit()


if __name__ == "__main__":
    main()