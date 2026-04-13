import torch
import torch.nn as nn
import torch.nn.functional as F


def get_graph_feature(x, k=20):
    B, C, N = x.size()
    x_trans = x.transpose(1, 2)
    dist = torch.cdist(x_trans, x_trans)
    idx = dist.topk(k=k, dim=-1, largest=False)[1]
    idx_base = torch.arange(0, B, device=x.device).view(-1, 1, 1) * N
    idx = (idx + idx_base).view(-1)
    feature = x_trans.contiguous().view(B * N, C)[idx, :]
    feature = feature.view(B, N, k, C).permute(0, 3, 1, 2).contiguous()
    x_expand = x.view(B, C, N, 1).expand(B, C, N, k)
    return torch.cat((feature - x_expand, x_expand), dim=1)


class EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, k=20):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x):
        x = get_graph_feature(x, self.k)
        x = self.conv(x)
        return x.max(dim=-1, keepdim=False)[0]


class DGCNNBBox(nn.Module):
    """
    Dynamic Graph CNN updated to accept 7 channels (XYZ + RGB + Mask).
    """

    def __init__(self, in_channels=7, k=20):
        super().__init__()
        self.k = k

        self.edge1 = EdgeConv(in_channels, 64, k)
        self.edge2 = EdgeConv(64, 64, k)
        self.edge3 = EdgeConv(64, 128, k)
        self.edge4 = EdgeConv(128, 256, k)

        self.conv5 = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop = nn.Dropout(p=0.4)

        self.fc_center = nn.Linear(256, 3)
        self.fc_dims = nn.Linear(256, 3)
        self.fc_rot = nn.Linear(256, 6)

        # Smart Init (Identity Rotation + ~13cm starting box)
        nn.init.zeros_(self.fc_center.weight)
        nn.init.zeros_(self.fc_center.bias)
        nn.init.zeros_(self.fc_dims.weight)
        self.fc_dims.bias.data = torch.tensor([-2.0, -2.0, -2.0])
        nn.init.zeros_(self.fc_rot.weight)
        self.fc_rot.bias.data = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])

    def forward(self, x):
        x1 = self.edge1(x)
        x2 = self.edge2(x1)
        x3 = self.edge3(x2)
        x4 = self.edge4(x3)

        x_cat = torch.cat((x1, x2, x3, x4), dim=1)
        x_out = self.conv5(x_cat)
        g = x_out.max(dim=2)[0]

        g = F.leaky_relu(self.bn1(self.fc1(g)), negative_slope=0.2)
        g = self.drop(g)
        g = F.leaky_relu(self.bn2(self.fc2(g)), negative_slope=0.2)
        g = self.drop(g)

        center_offset = self.fc_center(g)
        log_dims = torch.clamp(self.fc_dims(g), min=-3.5, max=2.0)
        rot6d = self.fc_rot(g)

        return center_offset, log_dims, rot6d

    def get_3d_box(self, center, log_dims, rot6d):
        B = center.shape[0]
        dims = torch.exp(log_dims) / 2.0

        unit = torch.tensor([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
        ], dtype=torch.float32, device=center.device).unsqueeze(0).repeat(B, 1, 1)

        corners = unit * dims.unsqueeze(1)

        x_raw = rot6d[:, 0:3]
        y_raw = rot6d[:, 3:6]

        x = F.normalize(x_raw, p=2, dim=1, eps=1e-6)
        y = y_raw - (x * y_raw).sum(dim=1, keepdim=True) * x
        y = F.normalize(y, p=2, dim=1, eps=1e-6)
        z = torch.cross(x, y, dim=1)

        R = torch.stack([x, y, z], dim=-1)
        rotated_corners = torch.bmm(corners, R.transpose(1, 2))

        return rotated_corners + center.unsqueeze(1)