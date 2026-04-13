"""
train.py  —  Training script  (v2)
====================================
Improvements over v1:
  1. Composite loss:
       • Chamfer distance  (order-agnostic corner matching, same as v1)
       • Corner L1 after Hungarian matching  (directly minimises MCD)
       • Smooth-L1 on centre offset          (stable early training)
       • T-Net orthogonality regularisation  (keeps feature T-Net valid)
  2. Cosine LR schedule with warm-up instead of ReduceLROnPlateau.
  3. Gradient clipping (max norm 1.0) to prevent exploding gradients.
  4. Automatic Mixed Precision (AMP) for speed on CUDA.
  5. Proper train/val split done once at startup (not re-shuffled per epoch).
  6. Best model selected on validation MCD (metres), not Chamfer loss.
  7. Saves full training log to CSV.
"""

import os
import csv
import math
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.amp import GradScaler, autocast
from scipy.optimize import linear_sum_assignment

from dataset import PointCloudInstanceDataset
from model   import PointNetBBox, tnet_reg_loss


# ── Hungarian-matched corner L1 ───────────────────────────────────────────────
def hungarian_corner_loss(pred, target):
    """
    pred, target : (B, 8, 3)
    Finds the best permutation of the 8 predicted corners to match GT
    (handles ambiguity in corner ordering), then computes mean Smooth-L1.
    Cast to float32 first — cdist_cuda doesn't support fp16.
    """
    pred   = pred.float()
    target = target.float()
    B      = pred.shape[0]
    loss   = 0.0
    for b in range(B):
        cost = torch.cdist(pred[b], target[b]).detach().cpu().numpy()  # (8,8)
        row, col = linear_sum_assignment(cost)
        loss += F.smooth_l1_loss(pred[b][row], target[b][col], beta=0.05)
    return loss / B


# ── mean corner distance (metric, no gradient) ────────────────────────────────
@torch.no_grad()
def mean_corner_dist(pred, target):
    """Mean L2 over matched corners, in metres.
    Cast to float32 — cdist_cuda doesn't support fp16."""
    pred   = pred.float()
    target = target.float()
    B      = pred.shape[0]
    mcds   = []
    for b in range(B):
        cost = torch.cdist(pred[b], target[b]).cpu().numpy()
        row, col = linear_sum_assignment(cost)
        d = (pred[b][row] - target[b][col]).pow(2).sum(-1).sqrt().mean()
        mcds.append(d.item())
    return mcds   # return list so caller can aggregate robustly


# ── cosine LR with linear warm-up ────────────────────────────────────────────
def cosine_lr(optimizer, epoch, warmup, total, base_lr, min_lr=1e-6):
    if epoch < warmup:
        lr = base_lr * (epoch + 1) / max(warmup, 1)
    else:
        t  = (epoch - warmup) / max(total - warmup, 1)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * t))
    for g in optimizer.param_groups:
        g["lr"] = lr
    return lr


# ── main ──────────────────────────────────────────────────────────────────────
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataset — build once, split by index
    full_ds = PointCloudInstanceDataset(args.data_root,
                                        num_points=args.num_points,
                                        is_train=True)
    n    = len(full_ds)
    idx  = np.random.permutation(n)
    split= int(0.30 * n)   # 30% val instead of 20% — more stable estimates
    train_idx, val_idx = idx[split:], idx[:split]

    train_ds = Subset(PointCloudInstanceDataset(
                    args.data_root, args.num_points, is_train=True),  train_idx)
    val_ds   = Subset(PointCloudInstanceDataset(
                    args.data_root, args.num_points, is_train=False), val_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=True)

    print(f"Train={len(train_ds)}  Val={len(val_ds)}")

    model     = PointNetBBox(in_channels=6, num_points=args.num_points).to(device)
    total_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_p/1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=args.lr, weight_decay=1e-4)
    scaler    = GradScaler("cuda", enabled=(device.type == "cuda"))

    # CSV log
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    log_path = args.save_path.replace(".pth", "_log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch","lr","train_loss","val_loss",
                                 "train_mcd_mean","val_mcd_mean",
                                 "train_mcd_median","val_mcd_median"])

    best_mcd = float("inf")

    for epoch in range(args.epochs):
        lr = cosine_lr(optimizer, epoch, args.warmup, args.epochs, args.lr)

        # ── train ──────────────────────────────────────────────────────────
        model.train()
        t_loss = 0.
        t_mcds = []   # collect per-sample MCDs for robust aggregation

        for batch in train_loader:
            pts  = batch["points"].to(device)
            tgt  = batch["target"].to(device).view(-1, 8, 3)

            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=(device.type=="cuda")):
                center, log_dims, rot6d, T_feat = model(pts)
                pred = model.get_3d_box(center, log_dims, rot6d)

                pred_f = pred.float()
                tgt_f  = tgt.float()

                dist      = torch.cdist(pred_f, tgt_f)
                chamfer   = dist.min(2)[0].mean() + dist.min(1)[0].mean()
                hungarian = hungarian_corner_loss(pred_f, tgt_f)
                tgt_centre  = tgt_f.mean(dim=1)
                centre_loss = F.smooth_l1_loss(center.float(), tgt_centre, beta=0.01)
                orth_loss   = tnet_reg_loss(T_feat) * 0.001
                loss = chamfer + 0.5 * hungarian + 0.2 * centre_loss + orth_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            t_loss += loss.item()
            t_mcds.extend(mean_corner_dist(pred_f, tgt_f))

        t_loss /= len(train_loader)
        t_mcd_mean   = float(np.mean(t_mcds))
        t_mcd_median = float(np.median(t_mcds))

        # ── validate ────────────────────────────────────────────────────────
        model.eval()
        v_loss = 0.
        v_mcds = []   # collect per-sample MCDs

        with torch.no_grad():
            for batch in val_loader:
                pts = batch["points"].to(device)
                tgt = batch["target"].to(device).view(-1, 8, 3)

                center, log_dims, rot6d, T_feat = model(pts)
                pred   = model.get_3d_box(center, log_dims, rot6d)
                pred_f = pred.float()
                tgt_f  = tgt.float()

                dist    = torch.cdist(pred_f, tgt_f)
                chamfer = dist.min(2)[0].mean() + dist.min(1)[0].mean()
                v_loss += chamfer.item()
                v_mcds.extend(mean_corner_dist(pred_f, tgt_f))

        v_loss /= len(val_loader)
        v_mcd_mean   = float(np.mean(v_mcds))
        v_mcd_median = float(np.median(v_mcds))

        print(f"Ep {epoch+1:03d}/{args.epochs}  lr={lr:.2e}  "
              f"train: loss={t_loss:.4f} MCD={t_mcd_mean:.4f}m (med={t_mcd_median:.4f}m)  "
              f"val:   loss={v_loss:.4f} MCD={v_mcd_mean:.4f}m (med={v_mcd_median:.4f}m)",
              flush=True)

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch+1, lr, t_loss, v_loss,
                                    t_mcd_mean, v_mcd_mean,
                                    t_mcd_median, v_mcd_median])

        # Save on best *median* val MCD — more robust than mean to outlier spikes
        if v_mcd_median < best_mcd:
            best_mcd = v_mcd_median
            torch.save({
                "epoch":       epoch+1,
                "model":       model.state_dict(),
                "optimizer":   optimizer.state_dict(),
                "best_mcd":    best_mcd,
                "args":        vars(args),
            }, args.save_path)
            print(f"  ✓ Best MCD {best_mcd:.4f} m — saved to {args.save_path}")

    print(f"\nTraining complete. Best val median MCD = {best_mcd:.4f} m")
    print(f"Log saved to {log_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",    required=True)
    p.add_argument("--epochs",       type=int,   default=80)
    p.add_argument("--batch_size",   type=int,   default=16)
    p.add_argument("--num_points",   type=int,   default=1024)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--warmup",       type=int,   default=5,
                   help="LR warm-up epochs")
    p.add_argument("--num_workers",  type=int,   default=8)
    p.add_argument("--save_path",    type=str,   default="best_model.pth")
    main(p.parse_args())