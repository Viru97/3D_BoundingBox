import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetBBox(nn.Module):
    """
    Pure 6D Pose Regression PointNet (No T-Net!).
    By removing the T-Net, the network maintains spatial awareness of the
    absolute rotation, which is strictly required for 3D Pose Estimation.
    """
    def __init__(self, in_channels=6, num_points=1024):
        super().__init__()
        self.in_channels = in_channels

        # ── Point feature extraction ────────────────────────────────────────
        self.conv1 = nn.Conv1d(in_channels, 64,  1)
        self.conv2 = nn.Conv1d(64,          128, 1)
        self.conv3 = nn.Conv1d(128,         256, 1)
        self.conv4 = nn.Conv1d(256,         1024, 1)

        self.bn1   = nn.BatchNorm1d(64)
        self.bn2   = nn.BatchNorm1d(128)
        self.bn3   = nn.BatchNorm1d(256)
        self.bn4   = nn.BatchNorm1d(1024)

        # ── Global regression MLP ───────────────────────────────────────────
        self.fc1   = nn.Linear(1024, 512)
        self.fc2   = nn.Linear(512,  256)
        self.fc3   = nn.Linear(256,  128)

        self.bn5   = nn.BatchNorm1d(512)
        self.bn6   = nn.BatchNorm1d(256)
        self.bn7   = nn.BatchNorm1d(128)

        self.drop1 = nn.Dropout(p=0.4)
        self.drop2 = nn.Dropout(p=0.4)

        # ── Disentangled output heads ───────────────────────────────────────
        self.fc_center = nn.Linear(128, 3)   # centre offset from anchor
        self.fc_dims   = nn.Linear(128, 3)   # log half-extents
        self.fc_rot    = nn.Linear(128, 6)   # 6D continuous rotation

        # ── SMART INITIALIZATION ────────────────────────────────────────────
        nn.init.zeros_(self.fc_center.weight)
        nn.init.zeros_(self.fc_center.bias)

        nn.init.zeros_(self.fc_dims.weight)
        # Log(-2.0) = 0.135m -> This starts the model at ~13cm boxes instead of 1 meter
        self.fc_dims.bias.data = torch.tensor([-2.0, -2.0, -2.0])

        nn.init.zeros_(self.fc_rot.weight)
        # Identity rotation (x=[1,0,0], y=[0,1,0]) to prevent NaN division in Gram-Schmidt
        self.fc_rot.bias.data = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])

    def forward(self, x):
        # x : (B, 6, N)
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.bn3(self.conv3(h)))
        h = F.relu(self.bn4(self.conv4(h)))

        # Global descriptors (max-pool)
        g = h.max(dim=2)[0]               # (B, 1024)

        # Regression MLP
        g = F.relu(self.bn5(self.fc1(g)))
        g = self.drop1(g)
        g = F.relu(self.bn6(self.fc2(g)))
        g = self.drop2(g)
        g = F.relu(self.bn7(self.fc3(g)))     # (B, 128)

        center_offset = self.fc_center(g)
        log_dims      = torch.clamp(self.fc_dims(g), min=-3.5, max=2.0)
        rot6d         = self.fc_rot(g)

        # Notice: Returning 3 items now!
        return center_offset, log_dims, rot6d

    def get_3d_box(self, center, log_dims, rot6d):
        B = center.shape[0]

        dims = torch.exp(log_dims) / 2.0       # (B, 3)

        unit = torch.tensor([
            [-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
            [-1,-1, 1],[1,-1, 1],[1,1, 1],[-1,1, 1],
        ], dtype=torch.float32, device=center.device)
        corners = unit.unsqueeze(0) * dims.unsqueeze(1)  # (B, 8, 3)

        # 6D → orthogonal rotation matrix (Gram-Schmidt)
        x_raw = rot6d[:, :3]
        y_raw = rot6d[:, 3:]
        x = F.normalize(x_raw, p=2, dim=1, eps=1e-6)
        y = F.normalize(y_raw - (x * y_raw).sum(1, keepdim=True) * x, p=2, dim=1, eps=1e-6)
        z = torch.cross(x, y, dim=1)
        R = torch.stack([x, y, z], dim=-1)     # (B, 3, 3)

        return torch.bmm(corners, R.transpose(1, 2)) + center.unsqueeze(1)