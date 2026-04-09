import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetBBox(nn.Module):
    """
    6D Pose Regression PointNet.
    Predicts Center, Dimensions, and 6D continuous rotation.
    Internally constructs a mathematically flawless rigid 3D bounding box.
    """

    def __init__(self, in_channels=6):
        super().__init__()

        # Point Feature Extraction
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        # Global Feature Regression
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)

        # Split Heads for explicit BBox parametrization
        self.fc_center = nn.Linear(256, 3)
        self.fc_dims = nn.Linear(256, 3)
        self.fc_rot = nn.Linear(256, 6)  # 6D continuous rotation representation

        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        # x shape: (Batch, 6, 1024)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))

        # Global Max Pooling
        x = torch.max(x, dim=2, keepdim=False)[0]

        # Regression
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn5(self.fc2(x)))

        center_offset = self.fc_center(x)
        log_dims = torch.clamp(self.fc_dims(x), min=-4.0, max=4.0)  # Keeps dims in [~0.009m, ~27m]
        rot6d = self.fc_rot(x)

        return center_offset, log_dims, rot6d

    def get_3d_box(self, center, log_dims, rot6d):
        """Deterministically generates the 8 corners of the rotated bounding box."""
        B = center.shape[0]
        dims = torch.exp(log_dims) / 2.0  # Half-extents

        # Canonical 8 corners of a unit box centered at origin
        corners = torch.tensor([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
        ], dtype=torch.float32, device=center.device).unsqueeze(0).repeat(B, 1, 1)

        # Scale to predicted dimensions
        corners = corners * dims.unsqueeze(1)  # (B, 8, 3)

        # 6D representation to Orthogonal 3x3 Rotation Matrix (Gram-Schmidt)
        x_raw = rot6d[:, 0:3]
        y_raw = rot6d[:, 3:6]

        x = F.normalize(x_raw, p=2, dim=1, eps=1e-6)
        y = y_raw - (x * y_raw).sum(dim=1, keepdim=True) * x
        y = F.normalize(y, p=2, dim=1, eps=1e-6)
        z = torch.cross(x, y, dim=1)

        R = torch.stack([x, y, z], dim=-1)  # (Batch, 3, 3)

        # Rotate and apply center translation
        rotated_corners = torch.bmm(corners, R.transpose(1, 2))
        final_corners = rotated_corners + center.unsqueeze(1)

        return final_corners