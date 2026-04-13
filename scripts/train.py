"""
train.py — Training script
===========================
Usage:
    PYTHONPATH=$PWD/src python scripts/train.py --data_root ~/Downloads/dl_challenge

Key design decisions:
  • Cosine LR with linear warm-up (peak = TrainConfig.learning_rate = 1e-3)
  • AdamW with weight_decay=1e-4
  • AMP (fp16) + gradient clipping
  • torch.compile() for CUDA graph optimisation
  • Best checkpoint selected on validation *median* MCD (robust to outlier batches)
"""

import os, csv, math, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.amp import GradScaler, autocast
from scipy.optimize import linear_sum_assignment

from sereact_bbox.dataset    import PointCloudInstanceDataset
from sereact_bbox.model      import DGCNNBBox
from sereact_bbox.loss       import AccuracyBoxLoss
from sereact_bbox.validation import create_splits
from sereact_bbox.config     import TrainConfig, ModelConfig, DataConfig

# Full float32 precision for rotation matrix accuracy
torch.set_float32_matmul_precision("highest")


def collate_fn(batch):
    """Drop the unused anchor field so DataLoader doesn't have to handle it."""
    return {
        "points": torch.stack([b["points"] for b in batch]),
        "target": torch.stack([b["target"] for b in batch]),
    }


@torch.no_grad()
def mean_corner_dist(pred, target):
    """Per-sample MCD list for robust median aggregation."""
    pred, target = pred.float(), target.float()
    mcds = []
    for b in range(pred.shape[0]):
        cost     = torch.cdist(pred[b], target[b]).cpu().numpy()
        row, col = linear_sum_assignment(cost)
        d        = (pred[b][row] - target[b][col]).pow(2).sum(-1).sqrt().mean()
        mcds.append(d.item())
    return mcds


def cosine_lr(optimizer, epoch, warmup, total, base_lr, min_lr=1e-6):
    if epoch < warmup:
        lr = base_lr * (epoch + 1) / max(warmup, 1)
    else:
        t  = (epoch - warmup) / max(total - warmup, 1)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * t))
    for g in optimizer.param_groups:
        g["lr"] = lr
    return lr


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Peak LR={TrainConfig.learning_rate:.0e} | "
          f"Batch={TrainConfig.batch_size} | TF32=disabled")

    # ── dataset ───────────────────────────────────────────────────────────
    full_ds  = PointCloudInstanceDataset(args.data_root,
                                         num_points=DataConfig.num_points,
                                         is_train=True)
    train_idx, val_idx, _ = create_splits(
        len(full_ds),
        DataConfig.is_train_split, DataConfig.val_split, DataConfig.test_split)

    train_ds = Subset(PointCloudInstanceDataset(
        args.data_root, DataConfig.num_points, is_train=True),  train_idx)
    val_ds   = Subset(PointCloudInstanceDataset(
        args.data_root, DataConfig.num_points, is_train=False), val_idx)

    lkw = dict(num_workers=TrainConfig.num_workers,
               pin_memory=True, collate_fn=collate_fn)
    train_loader = DataLoader(train_ds, batch_size=TrainConfig.batch_size,
                              shuffle=True, drop_last=True, **lkw)
    val_loader   = DataLoader(val_ds,   batch_size=TrainConfig.batch_size,
                              shuffle=False, **lkw)
    print(f"Train={len(train_ds)}  Val={len(val_ds)}")

    # ── model ─────────────────────────────────────────────────────────────
    model = DGCNNBBox(in_channels=ModelConfig.in_channels).to(device)
    if hasattr(torch, "compile"):
        try:
            print("Applying torch.compile()...")
            model = torch.compile(model)
        except Exception as e:
            print(f"torch.compile() skipped: {e}")

    criterion = AccuracyBoxLoss()
    optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=TrainConfig.learning_rate,
                                   weight_decay=TrainConfig.weight_decay)
    scaler    = GradScaler("cuda", enabled=(device.type == "cuda"))

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    log_path = args.save_path.replace(".pth", "_log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch","lr","train_loss","val_loss",
            "train_mcd_mean","val_mcd_mean",
            "train_mcd_median","val_mcd_median"])

    best_mcd = float("inf")

    for epoch in range(TrainConfig.epochs):
        lr = cosine_lr(optimizer, epoch,
                       TrainConfig.warmup_epochs, TrainConfig.epochs,
                       TrainConfig.learning_rate, TrainConfig.min_lr)

        # ── train ──────────────────────────────────────────────────────────
        model.train()
        t_loss, t_mcds = 0., []

        for batch in train_loader:
            pts = batch["points"].to(device)
            tgt = batch["target"].to(device).view(-1, 8, 3)

            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=(device.type == "cuda")):
                center, log_dims, rot6d = model(pts)
                pred = model.get_3d_box(center, log_dims, rot6d)
                loss = criterion(pred.float(), tgt.float(), center)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           TrainConfig.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            t_loss += loss.item()
            t_mcds.extend(mean_corner_dist(pred.float(), tgt.float()))

        t_loss /= len(train_loader)

        # ── validate ────────────────────────────────────────────────────────
        model.eval()
        v_loss, v_mcds = 0., []

        with torch.no_grad():
            for batch in val_loader:
                pts = batch["points"].to(device)
                tgt = batch["target"].to(device).view(-1, 8, 3)
                center, log_dims, rot6d = model(pts)
                pred = model.get_3d_box(center, log_dims, rot6d)
                loss = criterion(pred.float(), tgt.float(), center)
                v_loss += loss.item()
                v_mcds.extend(mean_corner_dist(pred.float(), tgt.float()))

        v_loss   /= len(val_loader)
        t_mean, t_med = np.mean(t_mcds), np.median(t_mcds)
        v_mean, v_med = np.mean(v_mcds), np.median(v_mcds)

        print(f"Ep {epoch+1:03d}/{TrainConfig.epochs}  lr={lr:.2e}  "
              f"train: loss={t_loss:.4f} MCD={t_mean:.4f}m (med={t_med:.4f}m)  "
              f"val:   loss={v_loss:.4f} MCD={v_mean:.4f}m (med={v_med:.4f}m)",
              flush=True)

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch+1, lr, t_loss, v_loss,
                t_mean, v_mean, t_med, v_med])

        if v_med < best_mcd:
            best_mcd = v_med
            torch.save({
                "epoch":    epoch+1,
                "model":    model.state_dict(),
                "optimizer":optimizer.state_dict(),
                "best_mcd": best_mcd,
            }, args.save_path)
            print(f"  ✓ Best MCD {best_mcd:.4f} m — saved to {args.save_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",  required=True)
    p.add_argument("--save_path",  default="best_model.pth")
    main(p.parse_args())