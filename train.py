import argparse
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from dataset import PointCloudInstanceDataset
from model import PointNetBBox


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset_train = PointCloudInstanceDataset(args.data_root, num_points=1024, is_train=True)
    dataset_val = PointCloudInstanceDataset(args.data_root, num_points=1024, is_train=False)

    dataset_size = len(dataset_train)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    split = int(np.floor(0.2 * dataset_size))
    train_indices, val_indices = indices[split:], indices[:split]

    train_loader = DataLoader(dataset_train, batch_size=args.batch_size,
                              sampler=torch.utils.data.SubsetRandomSampler(train_indices), num_workers=4)
    val_loader = DataLoader(dataset_val, batch_size=args.batch_size,
                            sampler=torch.utils.data.SubsetRandomSampler(val_indices), num_workers=4)

    model = PointNetBBox(in_channels=6).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            points = batch['points'].to(device)
            targets = batch['target'].to(device).view(-1, 8, 3)  # Ground Truth corners

            optimizer.zero_grad()

            # Forward pass retrieves BBox components and explicitly builds corners
            center_offset, log_dims, rot6d = model(points)
            preds = model.get_3d_box(center_offset, log_dims, rot6d)  # (B, 8, 3)

            # Chamfer Distance Loss: order-agnostic corner matching
            dist = torch.cdist(preds, targets)  # (B, 8, 8)
            loss = dist.min(dim=2)[0].mean() + dist.min(dim=1)[0].mean()

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                points = batch['points'].to(device)
                targets = batch['target'].to(device).view(-1, 8, 3)

                center_offset, log_dims, rot6d = model(points)
                preds = model.get_3d_box(center_offset, log_dims, rot6d)

                dist = torch.cdist(preds, targets)
                loss = dist.min(dim=2)[0].mean() + dist.min(dim=1)[0].mean()
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch + 1:02d}/{args.epochs} | Train Chamfer Dist: {train_loss:.4f} | Val Chamfer Dist: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("  -> Model saved!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    main(args)