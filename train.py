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
from model   import PointNetBBox

def hungarian_corner_loss(pred, target):
    pred   = pred.float()
    target = target.float()
    B      = pred.shape[0]
    loss   = 0.0
    for b in range(B):
        cost = torch.cdist(pred[b], target[b]).detach().cpu().numpy()
        row, col = linear_sum_assignment(cost)
        loss += F.smooth_l1_loss(pred[b][row], target[b][col], beta=0.05)
    return loss / B

@torch.no_grad()
def mean_corner_dist(pred, target):
    pred   = pred.float()
    target = target.float()
    B      = pred.shape[0]
    mcds   = []
    for b in range(B):
        cost = torch.cdist(pred[b], target[b]).cpu().numpy()
        row, col = linear_sum_assignment(cost)
        d = (pred[b][row] - target[b][col]).pow(2).sum(-1).sqrt().mean()
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
    print(f"Device: {device}")

    full_ds = PointCloudInstanceDataset(args.data_root, num_points=args.num_points, is_train=True)
    n    = len(full_ds)
    idx  = np.random.permutation(n)
    split= int(0.30 * n)
    train_idx, val_idx = idx[split:], idx[:split]

    train_ds = Subset(PointCloudInstanceDataset(args.data_root, args.num_points, is_train=True),  train_idx)
    val_ds   = Subset(PointCloudInstanceDataset(args.data_root, args.num_points, is_train=False), val_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model     = PointNetBBox(in_channels=6, num_points=args.num_points).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler    = GradScaler("cuda", enabled=(device.type == "cuda"))

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    log_path = args.save_path.replace(".pth", "_log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch","lr","train_loss","val_loss", "train_mcd_mean","val_mcd_mean", "train_mcd_median","val_mcd_median"])

    best_mcd = float("inf")

    for epoch in range(args.epochs):
        lr = cosine_lr(optimizer, epoch, args.warmup, args.epochs, args.lr)

        model.train()
        t_loss = 0.
        t_mcds = []

        for batch in train_loader:
            pts    = batch["points"].to(device)
            anchor = batch["anchor"].to(device)
            tgt    = batch["target"].to(device).view(-1, 8, 3)

            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=(device.type=="cuda")):
                # FIX: Removed T_feat unpacking
                center, log_dims, rot6d = model(pts)
                pred = model.get_3d_box(center, log_dims, rot6d)

                pred_f = pred.float()
                tgt_f  = tgt.float()

                loss_anchor = F.smooth_l1_loss(center.float(), anchor.float(), beta=0.01)

                dist      = torch.cdist(pred_f, tgt_f)
                chamfer   = dist.min(2)[0].mean() + dist.min(1)[0].mean()
                hungarian = hungarian_corner_loss(pred_f, tgt_f)

                # FIX: Removed orth_loss since there is no T-Net anymore
                loss = chamfer + 0.5 * hungarian + 0.5 * loss_anchor

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

        model.eval()
        v_loss = 0.
        v_mcds = []

        with torch.no_grad():
            for batch in val_loader:
                pts    = batch["points"].to(device)
                tgt    = batch["target"].to(device).view(-1, 8, 3)

                center, log_dims, rot6d = model(pts)
                pred   = model.get_3d_box(center, log_dims, rot6d)

                dist    = torch.cdist(pred.float(), tgt.float())
                chamfer = dist.min(2)[0].mean() + dist.min(1)[0].mean()
                v_loss += chamfer.item()
                v_mcds.extend(mean_corner_dist(pred.float(), tgt.float()))

        v_loss /= len(val_loader)
        v_mcd_mean   = float(np.mean(v_mcds))
        v_mcd_median = float(np.median(v_mcds))

        print(f"Ep {epoch+1:03d}/{args.epochs}  lr={lr:.2e}  "
              f"train: loss={t_loss:.4f} MCD={t_mcd_mean:.4f}m (med={t_mcd_median:.4f}m)  "
              f"val:   loss={v_loss:.4f} MCD={v_mcd_mean:.4f}m (med={v_mcd_median:.4f}m)", flush=True)

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch+1, lr, t_loss, v_loss, t_mcd_mean, v_mcd_mean, t_mcd_median, v_mcd_median])

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

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",    required=True)
    p.add_argument("--epochs",       type=int,   default=80)
    p.add_argument("--batch_size",   type=int,   default=16)
    p.add_argument("--num_points",   type=int,   default=2048)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--warmup",       type=int,   default=5)
    p.add_argument("--num_workers",  type=int,   default=8)
    p.add_argument("--save_path",    type=str,   default="best_model.pth")
    main(p.parse_args())