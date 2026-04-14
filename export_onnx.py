import os, argparse
import numpy as np
import torch
import torch.nn as nn
from model import DGCNNBBox

class OnnxWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        center, log_dims, rot6d = self.model(x)
        return center, log_dims, rot6d

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_points = 1024

    # Instantiate the 7-channel DGCNN
    model = DGCNNBBox(in_channels=7).to(device)

    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt.get("model", ckpt))
        print(f"Loaded weights from {args.checkpoint}")
    else:
        print("[WARN] Checkpoint not found. Exporting random initialization.")

    model.eval()
    wrapper = OnnxWrapper(model).to(device)

    # 7 channels for (X, Y, Z, R, G, B, Mask)
    dummy = torch.randn(1, 7, num_points, device=device)

    os.makedirs(args.out_dir, exist_ok=True)
    out_fp32 = os.path.join(args.out_dir, "pointnetbbox.onnx")

    print(f"\nExporting FP32 ONNX model (7-channels) to {out_fp32} ...")
    torch.onnx.export(
        wrapper, dummy, out_fp32,
        export_params=True,
        opset_version=14,
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
    p.add_argument("--checkpoint", default="best_model.pth")
    p.add_argument("--out_dir",    default="onnx_export")
    main(p.parse_args())