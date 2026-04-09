import argparse
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import RGBD3DDataset
from model import RGBDDetector3D


def focal_loss(pred, target):
    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()

    neg_weights = torch.pow(1 - target, 4)
    pred = torch.clamp(pred, 1e-4, 1 - 1e-4)

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    # Safely clamp num_pos to prevent NaNs if a batch happens to have 0 objects
    num_pos = torch.clamp(pos_inds.float().sum(), min=1.0)
    return -(pos_loss.sum() + neg_loss.sum()) / num_pos


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset_train = RGBD3DDataset(args.data_root, input_size=(512, 512), is_train=True)
    dataset_val = RGBD3DDataset(args.data_root, input_size=(512, 512), is_train=False)

    dataset_size = len(dataset_train)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    split = int(np.floor(0.2 * dataset_size))
    train_indices, val_indices = indices[split:], indices[:split]

    train_loader = DataLoader(dataset_train, batch_size=args.batch_size,
                              sampler=torch.utils.data.SubsetRandomSampler(train_indices), num_workers=4)
    val_loader = DataLoader(dataset_val, batch_size=args.batch_size,
                            sampler=torch.utils.data.SubsetRandomSampler(val_indices), num_workers=4)

    model = RGBDDetector3D().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)

    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]")
        for batch in pbar:
            inputs = batch['input'].to(device)
            hm_target = batch['heatmap'].to(device)
            corner_target = batch['corner_map'].to(device)
            reg_mask = batch['reg_mask'].to(device)

            optimizer.zero_grad()
            hm_pred, corner_pred = model(inputs)

            loss_hm = focal_loss(hm_pred, hm_target)
            mask_expand = reg_mask.expand_as(corner_pred)

            # CRITICAL FIX: Replaced Smooth L1 with strict L1 Loss.
            # Because metric offset errors are tiny (<1.0m), Smooth L1 squares them, causing vanishing gradients.
            loss_corner = F.l1_loss(corner_pred * mask_expand, corner_target * mask_expand, reduction='sum')
            loss_corner = loss_corner / (reg_mask.sum() * 24 + 1e-4)

            total_loss = loss_hm + args.weight_corner * loss_corner

            total_loss.backward()
            optimizer.step()
            train_loss += total_loss.item()
            pbar.set_postfix({'Loss': total_loss.item()})

        model.eval()
        val_loss, val_mae = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input'].to(device)
                hm_target = batch['heatmap'].to(device)
                corner_target = batch['corner_map'].to(device)
                reg_mask = batch['reg_mask'].to(device)

                hm_pred, corner_pred = model(inputs)

                loss_hm = focal_loss(hm_pred, hm_target)
                mask_expand = reg_mask.expand_as(corner_pred)

                # Use strict L1 Loss in validation too
                loss_corner = F.l1_loss(corner_pred * mask_expand, corner_target * mask_expand, reduction='sum') / (
                            reg_mask.sum() * 24 + 1e-4)

                val_loss += (loss_hm + args.weight_corner * loss_corner).item()

                abs_diff = torch.abs(corner_pred * mask_expand - corner_target * mask_expand)
                val_mae += (abs_diff.sum() / (reg_mask.sum() * 24 + 1e-4)).item()

        val_loss /= len(val_loader)
        val_mae /= len(val_loader)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch + 1} | Train Loss: {train_loss / len(train_loader):.4f} | Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.4f}m")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("-> Model saved!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_corner", type=float, default=20.0)
    args = parser.parse_args()
    main(args)