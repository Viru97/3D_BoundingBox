import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from bbox3d.config import cfg
from bbox3d.inference import load_model
from bbox3d.paths import fill_export_path_args, load_paths


class OnnxWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        center, log_dims, rot6d = self.model(x)
        corners = self.model.get_3d_box(center, log_dims, rot6d)
        return center, log_dims, rot6d, corners


def benchmark_ort(session, dummy_np, n=100, label=""):
    for _ in range(10):
        session.run(None, {"point_cloud": dummy_np})
    t0 = time.perf_counter()
    for _ in range(n):
        session.run(None, {"point_cloud": dummy_np})
    ms = (time.perf_counter() - t0) / n * 1000.0
    print(f"  {label:<22} {ms:.2f} ms / sample")
    return ms


def verify_parity(wrapper, onnx_path, dummy):
    import onnxruntime as ort

    with torch.no_grad():
        torch_out = [out.detach().cpu().numpy() for out in wrapper(dummy)]
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"point_cloud": dummy.cpu().numpy()})
    max_abs = max(float(np.max(np.abs(a - b))) for a, b in zip(torch_out, ort_out))
    if max_abs > 1e-4:
        raise RuntimeError(f"ONNX parity failed: max abs diff {max_abs:.6f}")
    print(f"  Verified PyTorch/ONNX parity: max abs diff {max_abs:.6f}")
    return sess


def main(args):
    fill_export_path_args(args, load_paths(args.paths_file))
    args.checkpoint = args.checkpoint or cfg.inference.weights
    args.out_dir = args.out_dir or "onnx_export"

    device = torch.device("cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, kwargs = load_model(
        args.checkpoint,
        device,
        allow_random_weights=args.allow_random_weights,
        model_version=args.model_version,
        in_channels=args.in_channels,
    )
    wrapper = OnnxWrapper(model).eval()
    dummy = torch.randn(1, kwargs["in_channels"], args.num_points, dtype=torch.float32)
    fp32_path = out_dir / "dgcnn_bbox.onnx"

    print(f"Exporting FP32 ONNX -> {fp32_path}")
    torch.onnx.export(
        wrapper,
        dummy,
        fp32_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
        input_names=["point_cloud"],
        output_names=["center", "log_dims", "rot6d", "corners"],
        dynamic_axes={
            "point_cloud": {0: "batch"},
            "center": {0: "batch"},
            "log_dims": {0: "batch"},
            "rot6d": {0: "batch"},
            "corners": {0: "batch"},
        },
    )
    print(f"  {os.path.getsize(fp32_path) / 1e6:.1f} MB")
    sess_fp32 = verify_parity(wrapper, fp32_path, dummy)

    if args.skip_int8:
        return

    print("INT8 dynamic quantization ...")
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        int8_path = out_dir / "dgcnn_bbox_int8.onnx"
        quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QUInt8)
        print(f"  {int8_path} ({os.path.getsize(int8_path) / 1e6:.1f} MB)")
        import onnxruntime as ort

        sess_int8 = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
        fp32_ms = benchmark_ort(sess_fp32, dummy.numpy(), label="ONNX FP32")
        int8_ms = benchmark_ort(sess_int8, dummy.numpy(), label="ONNX INT8")
        print(f"  Speedup: {fp32_ms / max(int8_ms, 1e-9):.2f}x")
    except Exception as exc:
        print(f"  INT8 failed: {exc}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--paths_file", default=None)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--num_points", type=int, default=cfg.data.num_points)
    p.add_argument("--model_version", choices=["v1", "v2"], default=None)
    p.add_argument("--in_channels", type=int, default=None)
    p.add_argument("--allow_random_weights", action="store_true")
    p.add_argument("--skip_int8", action="store_true")
    main(p.parse_args())
