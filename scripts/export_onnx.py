"""
export_onnx.py  —  Export DGCNNBBox to ONNX
================================================================
RESTORED: Correctly exports the 7-Channel DGCNN model using Opset 18
instead of crashing by trying to load the old Phase-1 2D ResNet.

Usage:
    python scripts/export_onnx.py --checkpoint best_model.pth --out_dir onnx_export
"""

import os, argparse
import torch
import torch.nn as nn
from sereact_bbox.model import DGCNNBBox
from sereact_bbox.config import DEFAULT_CHECKPOINT, DEFAULT_ONNX_DIR, ModelConfig, DataConfig


class OnnxWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        center, log_dims, rot6d = self.model(x)
        return center, log_dims, rot6d

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_points = DataConfig.num_points

    model = DGCNNBBox(in_channels=ModelConfig.in_channels).to(device)

    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model_state = ckpt.get("model", ckpt)
        # Handle uncompiled model loading into standard or compiled definitions easily
        model_state = {k.replace('_orig_mod.', ''): v for k, v in model_state.items()}
        model.load_state_dict(model_state, strict=False)
        print(f"Loaded weights from {args.checkpoint}")
    else:
        print("[WARN] Checkpoint not found. Exporting random initialization.")

    model.eval()
    wrapper = OnnxWrapper(model).to(device)
    wrapper.eval()

    # 7 channels for (X, Y, Z, R, G, B, Mask)
    dummy = torch.randn(1, ModelConfig.in_channels, num_points, device=device)

    os.makedirs(args.out_dir, exist_ok=True)
    out_fp32 = os.path.join(args.out_dir, "pointnetbbox.onnx")

    print(f"\nExporting FP32 ONNX model (7-channels) to {out_fp32} ...")
    torch.onnx.export(
        wrapper, dummy, out_fp32,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["point_cloud"],
        output_names=["center", "log_dims", "rot6d"],
        dynamic_axes={
            "point_cloud": {0: "batch_size", 2: "num_points"},
            "center":      {0: "batch_size"},
            "log_dims":    {0: "batch_size"},
            "rot6d":       {0: "batch_size"}
        }
    )

    print(f"Export successful. 7-Channel ONNX file saved to {out_fp32}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--out_dir",    default=DEFAULT_ONNX_DIR)
    main(p.parse_args())