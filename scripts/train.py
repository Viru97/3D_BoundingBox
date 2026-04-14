import os, csv, math, argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.amp import GradScaler, autocast
from scipy.optimize import linear_sum_assignment

from sereact_3d_bbox.config import cfg
from sereact_3d_bbox.data.dataset import PointCloudInstanceDataset
from sereact_3d_bbox.models.dgcnn import DGCNNBBox
from sereact_3d_bbox.models.loss import compute_loss, mean_corner_dist

def collate_fn(batch):
    return {
        "points": torch.stack([b["points"] for b in batch]),
        "target": torch.stack([b["target"] for b in batch]),
    }

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

    full_ds  = PointCloudInstanceDataset(args.data_root, args.num_points, is_train=True)
    n        = len(full_ds)
    rng      = np.random.default_rng(42)       # fixed seed -> reproducible splits
    idx      = rng.permutation(n)
    split    = int(0.30 * n)
    train_idx, val_idx = idx[split:].tolist(), idx[:split].tolist()

    train_ds = Subset(PointCloudInstanceDataset(args.data_root, args.num_points, True),  train_idx)
    val_ds   = Subset(PointCloudInstanceDataset(args.data_root, args.num_points, False), val_idx)

    lkw = dict(num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,  drop_last=True, **lkw)
    val_loader   = DataLoader(val_ds,   args.batch_size, shuffle=False, **lkw)
    print(f"Train={len(train_ds)}  Val={len(val_ds)}")

    model     = DGCNNBBox(in_channels=args.in_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler    = GradScaler("cuda", enabled=(device.type == "cuda"))

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    log_path = args.save_path.replace(".pth", "_log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch","lr","train_loss","val_loss",
                                 "train_mcd_mean","val_mcd_mean",
                                 "train_mcd_median","val_mcd_median"])
    best_mcd = float("inf")

    for epoch in range(args.epochs):
        lr = cosine_lr(optimizer, epoch, args.warmup, args.epochs, args.lr)

        model.train()
        t_loss, t_mcds = 0., []

        for batch in train_loader:
            pts = batch["points"].to(device)
            tgt = batch["target"].to(device).view(-1, 8, 3)

            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=(device.type == "cuda")):
                center, log_dims, rot6d = model(pts)
                pred   = model.get_3d_box(center, log_dims, rot6d)
                
                loss, chamfer, hungarian, loss_anchor = compute_loss(pred, tgt, center)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()

            t_loss += loss.item()
            t_mcds.extend(mean_corner_dist(pred, tgt))

        t_loss /= len(train_loader)

        model.eval()
        v_loss, v_mcds = 0., []
        with torch.no_grad():
            for batch in val_loader:
                pts = batch["points"].to(device)
                tgt = batch["target"].to(device).view(-1, 8, 3)
                center, log_dims, rot6d = model(pts)
                pred   = model.get_3d_box(center, log_dims, rot6d)
                
                dist    = torch.cdist(pred.float(), tgt.float())
                v_loss += dist.min(2)[0].mean() + dist.min(1)[0].mean()
                
                v_mcds.extend(mean_corner_dist(pred, tgt))

        v_loss    /= len(val_loader)
        t_mean, t_med = float(np.mean(t_mcds)), float(np.median(t_mcds))
        v_mean, v_med = float(np.mean(v_mcds)), float(np.median(v_mcds))

        print(f"Ep {epoch+1:03d}/{args.epochs}  lr={lr:.2e}  "
              f"train: loss={t_loss:.4f} MCD={t_mean:.4f}m (med={t_med:.4f}m)  "
              f"val:   loss={v_loss:.4f} MCD={v_mean:.4f}m (med={v_med:.4f}m)",
              flush=True)

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch+1, lr, t_loss, v_loss,
                                    t_mean, v_mean, t_med, v_med])

        if v_med < best_mcd:
            best_mcd = v_med
            torch.save({"epoch": epoch+1, "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "best_mcd": best_mcd, "args": vars(args)},
                       args.save_path)
            print(f"  ✓ Best MCD {best_mcd:.4f} m — saved to {args.save_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",   required=True)
    p.add_argument("--epochs",      type=int,   default=cfg.train.epochs)
    p.add_argument("--batch_size",  type=int,   default=cfg.train.batch_size)
    p.add_argument("--num_points",  type=int,   default=cfg.data.num_points)
    p.add_argument("--in_channels", type=int,   default=cfg.model.in_channels)
    p.add_argument("--lr",          type=float, default=cfg.train.learning_rate)
    p.add_argument("--warmup",      type=int,   default=5)
    p.add_argument("--num_workers", type=int,   default=8)
    p.add_argument("--save_path",   type=str,   default=cfg.train.save_path)
    main(p.parse_args())
