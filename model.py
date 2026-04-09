"""CenterNet3D — fixed: dtype mismatch in edges_to_corners."""

import torch
import torch.nn as nn
import torchvision.models as models
from config import Config


class DeconvNeck(nn.Module):
    def __init__(self, in_ch, out_ch=256):
        super().__init__()
        layers = []
        ch = in_ch
        for _ in range(3):
            layers += [
                nn.ConvTranspose2d(ch, out_ch, 4, stride=2, padding=1,
                                   bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            ch = out_ch
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)


class Head(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 1),
        )

    def forward(self, x):
        return self.net(x)


# Corner reconstruction signs (8 corners × 3 edges)

CORNER_SIGNS = torch.tensor([
    [-1, -1, -1],  # c0
    [+1, -1, -1],  # c1
    [+1, +1, -1],  # c2
    [-1, +1, -1],  # c3
    [-1, -1, +1],  # c4
    [+1, -1, +1],  # c5
    [+1, +1, +1],  # c6
    [-1, +1, +1],  # c7
], dtype=torch.float32) * 0.5  # (8, 3)


def edges_to_corners(edges):
    """Reconstruct 8 corner offsets from 3 edge vectors.

    Args:
        edges: (..., 9) — [e0_x, e0_y, e0_z, e1_x, ..., e2_z]
    Returns:
        corner_offsets: (..., 8, 3)
    """
    shape = edges.shape[:-1]
    e = edges.reshape(*shape, 3, 3)  # (..., 3_edges, 3_coords)

    # FIX: match dtype AND device of input tensor
    signs = CORNER_SIGNS.to(device=edges.device, dtype=edges.dtype)

    offsets = torch.einsum('kj, ...jd -> ...kd', signs, e)
    return offsets  # (..., 8, 3)


class CenterNet3D(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        nC = cfg.num_classes
        neck_ch = cfg.neck_channels
        head_ch = cfg.head_channels

        # ---- Backbone ----
        bb_map = {
            "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT, 512),
            "resnet34": (models.resnet34, models.ResNet34_Weights.DEFAULT, 512),
            "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT, 2048),
        }
        bb_cls, bb_weights, bb_out = bb_map[cfg.backbone]
        backbone = bb_cls(weights=bb_weights)

        if cfg.in_channels != 3:
            old = backbone.conv1
            backbone.conv1 = nn.Conv2d(
                cfg.in_channels, old.out_channels,
                kernel_size=old.kernel_size, stride=old.stride,
                padding=old.padding, bias=False,
            )
            with torch.no_grad():
                backbone.conv1.weight[:, :3] = old.weight
                for c in range(3, cfg.in_channels):
                    backbone.conv1.weight[:, c] = old.weight.mean(dim=1)

        self.stem   = nn.Sequential(backbone.conv1, backbone.bn1,
                                    backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # ---- Neck ----
        self.neck = DeconvNeck(bb_out, neck_ch)

        # ---- Heads ----
        self.hm_head     = Head(neck_ch, head_ch, nC)
        self.off2d_head  = Head(neck_ch, head_ch, 2)
        self.center_head = Head(neck_ch, head_ch, 3)
        self.edge_head   = Head(neck_ch, head_ch, 9)

        self._init_heads()

    def _init_heads(self):
        self.hm_head.net[-1].bias.data.fill_(-2.19)
        nn.init.zeros_(self.edge_head.net[-1].weight)
        nn.init.zeros_(self.edge_head.net[-1].bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        feat = self.neck(x)

        return {
            "heatmap":       torch.sigmoid(self.hm_head(feat)),
            "offset_2d":     self.off2d_head(feat),
            "center_offset": self.center_head(feat),
            "edges":         self.edge_head(feat),
        }