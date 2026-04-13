import os
import csv
import math
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.amp import GradScaler, autocast
from scipy.optimize import linear_sum_assignment

from sereact_bbox.dataset import PointCloudInstanceDataset
from sereact_bbox.model import DGCNNBBox
from sereact_bbox.validation import create_splits
from sereact_bbox.loss import AccuracyBoxLoss
from sereact_bbox.config import TrainConfig, ModelConfig, DataConfig

# RESTORED: Disable TF32 truncation. We need true 32-bit floats for millimeter-precise matrices!
torch.set_float32_matmul_precision('highest')


@torch.no_grad()
def mean_corner_dist(pred, target):
    pred = pred.float()
    target = target.float()
    B = pred.shape[0]
    mcds = []
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
        t = (epoch - warmup) / max(total - warmup, 1)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * t))
    for g in optimizer.param_groups:
        g["lr"] = lr
    return lr


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | TF32 Disabled (Max Precision Mode)")

    full_ds = PointCloudInstanceDataset(args.data_root, num_points=DataConfig.num_points, is_train=True)
    train_idx, val_idx, _ = create_splits(len(full_ds), DataConfig.is_train_split, DataConfig.val_split,
                                          DataConfig.test_split)

    train_ds = Subset(PointCloudInstanceDataset(args.data_root, DataConfig.num_points, is_train=True), train_idx)
    val_ds = Subset(PointCloudInstanceDataset(args.data_root, DataConfig.num_points, is_train=False), val_idx)

    train_loader = DataLoader(train_ds, batch_size=TrainConfig.batch_size, shuffle=True,
                              num_workers=TrainConfig.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=TrainConfig.batch_size, shuffle=False,
                            num_workers=TrainConfig.num_workers, pin_memory=True)

    model = DGCNNBBox(in_channels=ModelConfig.in_channels).to(device)

    if hasattr(torch, "compile"):
        try:
            print("Optimizing CUDA graph with torch.compile()...")
            model = torch.compile(model)
        except Exception as e:
            print(f"Skipping torch.compile...")

    criterion = AccuracyBoxLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=TrainConfig.learning_rate,
                                  weight_decay=TrainConfig.weight_decay)
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    log_path = args.save_path.replace(".pth", "_log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "lr", "train_loss", "val_loss", "train_mcd_mean", "val_mcd_mean", "train_mcd_median",
             "val_mcd_median"])

    best_mcd = float("inf")

    for epoch in range(TrainConfig.epochs):
        lr = cosine_lr(optimizer, epoch, TrainConfig.warmup_epochs, TrainConfig.epochs, TrainConfig.learning_rate)

        model.train()
        t_loss = 0.
        t_mcds = []

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), TrainConfig.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            t_loss += loss.item()
            t_mcds.extend(mean_corner_dist(pred.float(), tgt.float()))

        t_loss /= len(train_loader)
        t_mcd_mean = float(np.mean(t_mcds))
        t_mcd_median = float(np.median(t_mcds))

        model.eval()
        v_loss = 0.
        v_mcds = []

        with torch.no_grad():
            for batch in val_loader:
                pts = batch["points"].to(device)
                tgt = batch["target"].to(device).view(-1, 8, 3)

                center, log_dims, rot6d = model(pts)
                pred = model.get_3d_box(center, log_dims, rot6d)

                loss = criterion(pred.float(), tgt.float(), center)

                v_loss += loss.item()
                v_mcds.extend(mean_corner_dist(pred.float(), tgt.float()))

        v_loss /= len(val_loader)
        v_mcd_mean = float(np.mean(v_mcds))
        v_mcd_median = float(np.median(v_mcds))

        print(f"Ep {epoch + 1:03d}/{TrainConfig.epochs}  lr={lr:.2e}  "
              f"train: loss={t_loss:.4f} MCD={t_mcd_mean:.4f}m (med={t_mcd_median:.4f}m)  "
              f"val:   loss={v_loss:.4f} MCD={v_mcd_mean:.4f}m (med={v_mcd_median:.4f}m)", flush=True)

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, lr, t_loss, v_loss, t_mcd_mean, v_mcd_mean, t_mcd_median, v_mcd_median])

        if v_mcd_median < best_mcd:
            best_mcd = v_mcd_median
            torch.save({
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_mcd": best_mcd,
            }, args.save_path)
            print(f"  ✓ Best MCD {best_mcd:.4f} m — saved to {args.save_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--save_path", type=str, default="best_model.pth")
    p.add_argument("--epochs", type=int, default=TrainConfig.epochs, help="Number of training epochs")
    p.add_argument("--batch_size", type=int, default=TrainConfig.batch_size, help="Batch size for training")
    p.add_argument("--in_channels", type=int, default=ModelConfig.in_channels, help="Number of input channels")

    args = p.parse_args()

    # Override configs with args if provided
    TrainConfig.epochs = args.epochs
    TrainConfig.batch_size = args.batch_size
    ModelConfig.in_channels = args.in_channels

    main(args)