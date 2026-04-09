#!/usr/bin/env python3
"""Export to ONNX + FP16 + benchmark."""

import argparse
import os
import time
import numpy as np
import torch
import onnx
from config import Config
from model import CenterNet3D


class ExportWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return (out["heatmap"], out["offset_2d"],
                out["center_offset"], out["edges"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output",     type=str, default="centernet3d.onnx")
    p.add_argument("--opset",      type=int, default=14)
    p.add_argument("--fp16",       action="store_true")
    p.add_argument("--benchmark",  action="store_true")
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt.get("cfg", Config())
    model = CenterNet3D(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()

    wrapper = ExportWrapper(model)
    dummy = torch.randn(1, cfg.in_channels, cfg.input_height, cfg.input_width)

    torch.onnx.export(
        wrapper, dummy, args.output,
        input_names=["input"],
        output_names=["heatmap", "offset_2d", "center_offset", "edges"],
        dynamic_axes={n: {0: "batch"} for n in
                      ["input", "heatmap", "offset_2d", "center_offset", "edges"]},
        opset_version=args.opset, do_constant_folding=True)

    onnx_model = onnx.load(args.output)
    onnx.checker.check_model(onnx_model)
    mb = os.path.getsize(args.output) / 1e6
    print(f"✓ ONNX: {args.output} ({mb:.1f} MB)")

    try:
        import onnxsim
        opt, ok = onnxsim.simplify(onnx_model)
        if ok:
            onnx.save(opt, args.output)
            print(f"  ✓ Simplified")
    except ImportError:
        pass

    if args.fp16:
        try:
            from onnxconverter_common import float16
            m16 = float16.convert_float_to_float16(onnx.load(args.output))
            fp16_path = args.output.replace(".onnx", "_fp16.onnx")
            onnx.save(m16, fp16_path)
            print(f"  ✓ FP16: {fp16_path}")
        except ImportError:
            print("  pip install onnxconverter-common for FP16")

    if args.benchmark:
        import onnxruntime as ort
        sess = ort.InferenceSession(args.output,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        shape = [s if isinstance(s, int) else 1 for s in inp.shape]
        d = np.random.randn(*shape).astype(np.float32)
        for _ in range(10):
            sess.run(None, {inp.name: d})
        t0 = time.time()
        for _ in range(50):
            sess.run(None, {inp.name: d})
        ms = (time.time() - t0) / 50 * 1000
        print(f"  Latency: {ms:.1f}ms | FPS: {1000/ms:.1f}")


if __name__ == "__main__":
    main()