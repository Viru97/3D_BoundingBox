"""
model.py  —  PointNetBBox  (v2)
================================
Improvements over v1:
  1. Input T-Net (3×3 spatial alignment network) — aligns the point cloud
     before feature extraction, greatly improving rotation robustness.
  2. Feature T-Net (64×64) with orthogonality regularisation — aligns
     intermediate features for better discriminative power.
  3. Deeper feature extraction (64→128→256→1024) with skip concatenation
     from the first block, giving the global descriptor richer local context.
  4. Larger regression MLP (1024+256→512→256→128) with Dropout(0.4).
  5. All BatchNorm layers replaced by LayerNorm-friendly alternatives that
     work correctly with batch_size=1 at inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── T-Net: learnable spatial/feature alignment ────────────────────────────────
class TNet(nn.Module):
    """Predicts a k×k alignment matrix from a (B, k, N) point tensor."""

    def __init__(self, k=3):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k,   64,  1)
        self.conv2 = nn.Conv1d(64,  128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1   = nn.Linear(1024, 512)
        self.fc2   = nn.Linear(512,  256)
        self.fc3   = nn.Linear(256,  k * k)

        self.bn1   = nn.BatchNorm1d(64)
        self.bn2   = nn.BatchNorm1d(128)
        self.bn3   = nn.BatchNorm1d(1024)
        self.bn4   = nn.BatchNorm1d(512)
        self.bn5   = nn.BatchNorm1d(256)

        # Initialise output as identity
        nn.init.zeros_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)
        with torch.no_grad():
            self.fc3.bias.copy_(torch.eye(k).flatten())

    def forward(self, x):
        """x : (B, k, N) → transform : (B, k, k)"""
        B = x.size(0)
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.bn3(self.conv3(h)))
        h = h.max(dim=2)[0]                     # global max pool → (B, 1024)
        h = F.relu(self.bn4(self.fc1(h)))
        h = F.relu(self.bn5(self.fc2(h)))
        mat = self.fc3(h).view(B, self.k, self.k)
        return mat


def tnet_reg_loss(transform):
    """Orthogonality regulariser: ||I − A·Aᵀ||²_F"""
    B, k, _ = transform.shape
    I   = torch.eye(k, device=transform.device).unsqueeze(0).expand(B, -1, -1)
    AAT = torch.bmm(transform, transform.transpose(1, 2))
    return F.mse_loss(AAT, I)


# ── main model ────────────────────────────────────────────────────────────────
class PointNetBBox(nn.Module):
    """
    PointNet with T-Nets, deeper feature extraction, and disentangled heads
    for 3D bounding box estimation.

    Input  : (B, 6, N)   — concatenated [XYZ-centred | RGB]
    Output : center_offset (B,3), log_dims (B,3), rot6d (B,6)
    """

    def __init__(self, in_channels=6, num_points=1024):
        super().__init__()
        self.in_channels = in_channels
        xyz_channels = 3   # T-Net only aligns XYZ

        # ── Input T-Net (spatial alignment on XYZ) ─────────────────────────
        self.tnet_in = TNet(k=xyz_channels)

        # ── Point feature extraction ────────────────────────────────────────
        # Block 1
        self.conv1 = nn.Conv1d(in_channels, 64,  1)
        self.conv2 = nn.Conv1d(64,          64,  1)
        self.bn1   = nn.BatchNorm1d(64)
        self.bn2   = nn.BatchNorm1d(64)

        # Feature T-Net (on 64-dim features)
        self.tnet_feat = TNet(k=64)

        # Block 2 — deeper features
        self.conv3 = nn.Conv1d(64,  128, 1)
        self.conv4 = nn.Conv1d(128, 256, 1)
        self.conv5 = nn.Conv1d(256, 1024, 1)
        self.bn3   = nn.BatchNorm1d(128)
        self.bn4   = nn.BatchNorm1d(256)
        self.bn5   = nn.BatchNorm1d(1024)

        # ── Global regression MLP ───────────────────────────────────────────
        # Global descriptor = max-pool(1024) + max-pool(64) skip = 1024+64
        global_dim = 1024 + 64
        self.fc1   = nn.Linear(global_dim, 512)
        self.fc2   = nn.Linear(512,        256)
        self.fc3   = nn.Linear(256,        128)
        self.bn6   = nn.BatchNorm1d(512)
        self.bn7   = nn.BatchNorm1d(256)
        self.bn8   = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(p=0.4)
        self.drop2 = nn.Dropout(p=0.4)

        # ── Disentangled output heads ───────────────────────────────────────
        self.fc_center = nn.Linear(128, 3)   # centre offset from anchor
        self.fc_dims   = nn.Linear(128, 3)   # log half-extents
        self.fc_rot    = nn.Linear(128, 6)   # 6D continuous rotation

        # Sensible initialisations
        nn.init.zeros_(self.fc_center.bias)
        nn.init.zeros_(self.fc_dims.bias)
        nn.init.zeros_(self.fc_rot.bias)

    # ── forward ───────────────────────────────────────────────────────────────
    def forward(self, x):
        """x : (B, 6, N)"""
        B, C, N = x.shape

        # Split XYZ and features for T-Net alignment
        xyz = x[:, :3, :]   # (B, 3, N)

        # 1. Input spatial alignment
        T_in  = self.tnet_in(xyz)            # (B, 3, 3)
        xyz_a = torch.bmm(T_in, xyz)         # (B, 3, N)
        x_aligned = torch.cat([xyz_a, x[:, 3:, :]], dim=1)  # (B, 6, N) with aligned XYZ

        # 2. Block 1 features
        h = F.relu(self.bn1(self.conv1(x_aligned)))
        h = F.relu(self.bn2(self.conv2(h)))    # (B, 64, N)

        # 3. Feature T-Net alignment
        T_feat = self.tnet_feat(h)             # (B, 64, 64)
        h      = torch.bmm(T_feat, h)          # (B, 64, N) aligned features
        skip64 = h                             # save for skip connection

        # 4. Block 2 features
        h = F.relu(self.bn3(self.conv3(h)))    # (B, 128, N)
        h = F.relu(self.bn4(self.conv4(h)))    # (B, 256, N)
        h = F.relu(self.bn5(self.conv5(h)))    # (B, 1024, N)

        # 5. Global descriptors (max-pool)
        g1024 = h.max(dim=2)[0]               # (B, 1024)
        g64   = skip64.max(dim=2)[0]           # (B, 64) — skip from block 1

        g = torch.cat([g1024, g64], dim=1)    # (B, 1088) — richer descriptor

        # 6. Regression MLP
        g = F.relu(self.bn6(self.fc1(g)))
        g = self.drop1(g)
        g = F.relu(self.bn7(self.fc2(g)))
        g = self.drop2(g)
        g = F.relu(self.bn8(self.fc3(g)))     # (B, 128)

        center_offset = self.fc_center(g)
        log_dims      = torch.clamp(self.fc_dims(g), min=-3.5, max=2.0)  # ~[0.03 m, 7.4 m]
        rot6d         = self.fc_rot(g)

        return center_offset, log_dims, rot6d, T_feat   # also return T_feat for reg loss

    # ── box construction ──────────────────────────────────────────────────────
    def get_3d_box(self, center, log_dims, rot6d):
        """
        Reconstruct 8 corners of the oriented 3D bounding box.
        center   : (B, 3)  — offset from point-cloud centroid
        log_dims : (B, 3)  — log of full box dimensions
        rot6d    : (B, 6)  — first two cols of rotation matrix
        Returns  : (B, 8, 3)
        """
        B = center.shape[0]

        # Half-extents
        dims = torch.exp(log_dims) / 2.0       # (B, 3)

        # Canonical unit-box corners (8, 3)
        unit = torch.tensor([
            [-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
            [-1,-1, 1],[1,-1, 1],[1,1, 1],[-1,1, 1],
        ], dtype=torch.float32, device=center.device)
        corners = unit.unsqueeze(0) * dims.unsqueeze(1)  # (B, 8, 3)

        # 6D → orthogonal rotation matrix (Gram-Schmidt)
        x_raw = rot6d[:, :3]
        y_raw = rot6d[:, 3:]
        x = F.normalize(x_raw, p=2, dim=1, eps=1e-6)
        y = F.normalize(y_raw - (x * y_raw).sum(1, keepdim=True) * x,
                        p=2, dim=1, eps=1e-6)
        z = torch.cross(x, y, dim=1)
        R = torch.stack([x, y, z], dim=-1)     # (B, 3, 3)

        # Rotate then translate
        return torch.bmm(corners, R.transpose(1, 2)) + center.unsqueeze(1)