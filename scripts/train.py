import argparse
import csv
import json
import math
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from bbox3d.config import cfg
from bbox3d.data.dataset import PointCloudInstanceDataset, collate_fn
from bbox3d.data.splits import resolve_split_manifest, split_scene_ids
from bbox3d.metrics import mean_corner_distance
from bbox3d.models.dgcnn import build_model
from bbox3d.models.loss import BBoxLoss
from bbox3d.paths import fill_missing_path_args, load_paths


CHECKPOINT_FORMAT_VERSION = 1


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


def cosine_lr(optimizer, epoch, warmup, total, base_lr, min_lr):
    if epoch < warmup:
        lr = base_lr * (epoch + 1) / max(warmup, 1)
    else:
        t = (epoch - warmup) / max(total - warmup, 1)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * t))
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def atomic_torch_save(payload, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def checkpoint_payload(model, optimizer, scaler, epoch, best_mcd, args, manifest):
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "best_mcd": best_mcd,
        "args": vars(args),
        "config": asdict(cfg),
        "split_manifest": manifest,
    }


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    loss_sums = {}
    mcds = []

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            pts = batch["points"].to(device, non_blocking=True)
            tgt = batch["target"].to(device, non_blocking=True).float()
            target_center = batch["target_center"].to(device, non_blocking=True).float()
            target_dims = batch["target_dims"].to(device, non_blocking=True).float()

            if training:
                optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=device.type == "cuda"):
                center, log_dims, rot6d = model(pts)
                pred = model.get_3d_box(center, log_dims, rot6d)
                loss, components = criterion(
                    pred,
                    tgt,
                    center,
                    log_dims,
                    rot6d,
                    target_center=target_center,
                    target_dims=target_dims,
                )

            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                scaler.step(optimizer)
                scaler.update()

            total_loss += float(loss.detach().cpu())
            for name, value in components.items():
                loss_sums[name] = loss_sums.get(name, 0.0) + float(value.detach().cpu())
            mcds.extend(mean_corner_distance(pred.detach().float(), tgt.detach().float()))

    denom = max(len(loader), 1)
    losses = {name: value / denom for name, value in loss_sums.items()}
    return total_loss / denom, losses, float(np.mean(mcds)), float(np.median(mcds))


def main(args):
    fill_missing_path_args(args, load_paths(args.paths_file))
    if not args.data_root:
        raise SystemExit("Set data_root in paths.local.json or pass --data_root.")
    args.save_path = args.save_path or cfg.train.save_path

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    manifest = resolve_split_manifest(args.data_root, args.split_manifest, seed=args.seed, overwrite=args.rebuild_splits)
    train_scenes = split_scene_ids(manifest, "train")
    val_scenes = split_scene_ids(manifest, "val")

    train_ds = PointCloudInstanceDataset(args.data_root, args.num_points, True, train_scenes, seed=args.seed)
    val_ds = PointCloudInstanceDataset(args.data_root, args.num_points, False, val_scenes, seed=args.seed)
    if train_ds.skipped or val_ds.skipped:
        print(f"Skipped unusable instances: train={len(train_ds.skipped)} val={len(val_ds.skipped)}")

    loader_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
        worker_init_fn=worker_init,
    )
    train_loader = DataLoader(
        train_ds,
        args.batch_size,
        shuffle=True,
        drop_last=len(train_ds) >= args.batch_size,
        **loader_kwargs,
    )
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, drop_last=False, **loader_kwargs)
    print(f"Train scenes={len(train_scenes)} instances={len(train_ds)}")
    print(f"Val scenes={len(val_scenes)} instances={len(val_ds)}")

    model = build_model(args.model_version, args.in_channels, args.k_neighbors, args.dropout).to(device)
    criterion = BBoxLoss(
        chamfer_weight=args.chamfer_weight,
        corner_weight=args.corner_weight,
        center_weight=args.center_weight,
        dimension_weight=args.dimension_weight,
        rotation_weight=args.rotation_weight,
        rotation_regularizer_weight=args.rotation_regularizer_weight,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler("cuda", enabled=device.type == "cuda")

    start_epoch = 0
    best_mcd = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt.get("epoch", 0))
        best_mcd = float(ckpt.get("best_mcd", best_mcd))
        print(f"Resumed {args.resume} at epoch {start_epoch}")

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    last_save_path = Path(args.last_save_path) if args.last_save_path else save_path.with_name(save_path.stem + "_last.pth")
    log_path = save_path.with_name(save_path.stem + "_log.csv")
    write_header = not log_path.exists() or start_epoch == 0
    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(
                [
                    "epoch",
                    "lr",
                    "train_loss",
                    "val_loss",
                    "train_mcd_mean",
                    "val_mcd_mean",
                    "train_mcd_median",
                    "val_mcd_median",
                    "loss_components",
                ]
            )

    for epoch in range(start_epoch, args.epochs):
        lr = cosine_lr(optimizer, epoch, args.warmup, args.epochs, args.lr, args.min_lr)
        train_loss, train_components, train_mean, train_med = run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer, scaler=scaler
        )
        val_loss, val_components, val_mean, val_med = run_epoch(model, val_loader, criterion, device)

        print(
            f"Ep {epoch + 1:03d}/{args.epochs} lr={lr:.2e} "
            f"train loss={train_loss:.4f} MCD={train_mean:.4f}m med={train_med:.4f}m "
            f"val loss={val_loss:.4f} MCD={val_mean:.4f}m med={val_med:.4f}m",
            flush=True,
        )
        with log_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [
                    epoch + 1,
                    lr,
                    train_loss,
                    val_loss,
                    train_mean,
                    val_mean,
                    train_med,
                    val_med,
                    json.dumps({"train": train_components, "val": val_components}, sort_keys=True),
                ]
            )

        if val_med < best_mcd:
            best_mcd = val_med
            atomic_torch_save(
                checkpoint_payload(model, optimizer, scaler, epoch + 1, best_mcd, args, manifest),
                save_path,
            )
            print(f"  saved best median MCD {best_mcd:.4f} m -> {save_path}")
        atomic_torch_save(
            checkpoint_payload(model, optimizer, scaler, epoch + 1, best_mcd, args, manifest),
            last_save_path,
        )
        print(f"  saved resumable checkpoint -> {last_save_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--paths_file", default=None)
    p.add_argument("--data_root", default=None)
    p.add_argument("--split_manifest", default=None)
    p.add_argument("--rebuild_splits", action="store_true")
    p.add_argument("--epochs", type=int, default=cfg.train.epochs)
    p.add_argument("--batch_size", type=int, default=cfg.train.batch_size)
    p.add_argument("--num_points", type=int, default=cfg.data.num_points)
    p.add_argument("--in_channels", type=int, default=cfg.model.in_channels)
    p.add_argument("--model_version", default=cfg.model.version, choices=["v1", "v2"])
    p.add_argument("--k_neighbors", type=int, default=cfg.model.k_neighbors)
    p.add_argument("--dropout", type=float, default=cfg.model.dropout)
    p.add_argument("--lr", type=float, default=cfg.train.learning_rate)
    p.add_argument("--weight_decay", type=float, default=cfg.train.weight_decay)
    p.add_argument("--warmup", type=int, default=cfg.train.warmup_epochs)
    p.add_argument("--min_lr", type=float, default=cfg.train.min_lr)
    p.add_argument("--num_workers", type=int, default=cfg.train.num_workers)
    p.add_argument("--seed", type=int, default=cfg.train.seed)
    p.add_argument("--resume", default=cfg.train.resume)
    p.add_argument("--save_path", default=None)
    p.add_argument("--last_save_path", default=None)
    p.add_argument("--chamfer_weight", type=float, default=cfg.train.chamfer_weight)
    p.add_argument("--corner_weight", type=float, default=cfg.train.corner_weight)
    p.add_argument("--center_weight", type=float, default=cfg.train.center_weight)
    p.add_argument("--dimension_weight", type=float, default=cfg.train.dimension_weight)
    p.add_argument("--rotation_weight", type=float, default=cfg.train.rotation_weight)
    p.add_argument("--rotation_regularizer_weight", type=float, default=cfg.train.rotation_regularizer_weight)
    main(p.parse_args())
