import torch
import torch.nn as nn
from torchvision.models import resnet34


class RGBDDetector3D(nn.Module):
    """
    Early-Fusion RGB+Depth Network with Feature Pyramid Decoder.
    Takes 6-channel input and outputs a heatmap and 3D Bounding Box map.
    """

    def __init__(self, num_classes=1, num_corners=24):
        super(RGBDDetector3D, self).__init__()

        # 1. Base Encoder
        self.backbone = resnet34(pretrained=True)

        # Modify first layer to accept 6 channels (RGB + XYZ Point Cloud)
        old_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            # Copy RGB weights, initialize PC weights symmetrically
            self.backbone.conv1.weight[:, :3] = old_conv.weight
            self.backbone.conv1.weight[:, 3:] = old_conv.weight.mean(dim=1, keepdim=True)

        # 2. Decoder (Upsampling with skip connections to restore resolution)
        # ResNet layer features spatial sizes relative to input (e.g., 512x512):
        # layer1: /4 (128x128), layer2: /8 (64x64), layer3: /16 (32x32), layer4: /32 (16x16)

        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)  # -> /16
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # -> /8
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # -> /4

        # 3. Detection Heads
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()  # Probability map [0, 1]
        )

        self.corner_head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_corners, kernel_size=1, stride=1, padding=0)  # 24 coordinates
        )

    def forward(self, x):
        # Encoder
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x1 = self.backbone.layer1(x)  # stride 4
        x2 = self.backbone.layer2(x1)  # stride 8
        x3 = self.backbone.layer3(x2)  # stride 16
        x4 = self.backbone.layer4(x3)  # stride 32

        # Decoder with skip connections
        u1 = torch.relu(self.up1(x4))
        u2 = torch.relu(self.up2(u1 + x3))
        u3 = torch.relu(self.up3(u2 + x2))

        # Heads (Output stride is 4)
        hm = self.heatmap_head(u3)
        corners = self.corner_head(u3)

        return hm, corners