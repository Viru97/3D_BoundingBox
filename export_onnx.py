"""
export_onnx.py  —  Export PointNetBBox to ONNX + INT8 quantisation
===================================================================
Produces:
  • pointnetbbox.onnx         — FP32, runs on any ONNX Runtime backend
  • pointnetbbox_int8.onnx    — INT8 dynamically quantised (~4× smaller)

Usage:
    python export_onnx.py --checkpoint best_model.pth --out_dir onnx_export

Deployment:
    # ONNX Runtime (CPU / GPU)
    import onnxruntime as ort
    sess = ort.InferenceSession("pointnetbbox.onnx",
               providers=["CUDAExecutionProvider"])
    out  = sess.run(None, {"point_cloud": input_np})
    # out = [center (B,3), log_dims (B,3), rot6d (B,6)]

    # TensorRT (after ONNX export)
    trtexec --onnx=pointnetbbox.onnx --fp16 --saveEngine=pointnetbbox.trt
"""

import os, argparse
import numpy as np
import torch
import torch.nn as nn
from model import PointNetBBox


# ── wrapper: returns flat tuple (ONNX exporter needs tuple, not list) ─────────
class OnnxWrapper(nn.Module):
    """Wraps PointNetBBox so ONNX gets (center, log_dims, rot6d) as a flat tuple.
    T_feat (the internal alignment matrix) is dropped — not needed at inference.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        center, log_dims, rot6d, _ = self.model(x)
        return center, log_dims, rot6d


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  default="best_model.pth")
    p.add_argument("--out_dir",     default="onnx_export")
    p.add_argument("--num_points",  type=int, default=1024)
    p.add_argument("--batch_size",  type=int, default=1)
    p.add_argument("--opset",       type=int, default=18)
    p.add_argument("--no_quantise", action="store_true")
    return p.parse_args()


def main():
    args   = get_args()
    device = torch.device("cpu")    # export on CPU for maximum portability
    os.makedirs(args.out_dir, exist_ok=True)

    # ── load model ────────────────────────────────────────────────────────
    ckpt  = torch.load(args.checkpoint, map_location="cpu")
    saved = ckpt.get("args", {})
    num_points = saved.get("num_points", args.num_points)

    model = PointNetBBox(in_channels=6, num_points=num_points)
    model.load_state_dict(ckpt["model"])
    model.eval()

    wrapper = OnnxWrapper(model)
    wrapper.eval()

    dummy   = torch.randn(args.batch_size, 6, num_points)
    onnx_path = os.path.join(args.out_dir, "pointnetbbox.onnx")

    print(f"Exporting FP32 ONNX  (opset={args.opset}) …")
    print(f"  Input  : (batch={args.batch_size}, channels=6, points={num_points})")

    torch.onnx.export(
        wrapper, dummy, onnx_path,
        opset_version      = args.opset,
        input_names        = ["point_cloud"],
        output_names       = ["center", "log_dims", "rot6d"],
        dynamic_axes       = {
            "point_cloud": {0: "batch"},
            "center":      {0: "batch"},
            "log_dims":    {0: "batch"},
            "rot6d":       {0: "batch"},
        },
        do_constant_folding = True,
    )
    fp32_mb = os.path.getsize(onnx_path) / 1e6
    print(f"  ✓  {onnx_path}  ({fp32_mb:.1f} MB)")

    # ── verify with onnxruntime ───────────────────────────────────────────
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path,
               providers=["CPUExecutionProvider"])
        outs = sess.run(None, {"point_cloud": dummy.numpy()})
        print(f"  ✓  ONNXRuntime verification:")
        for name, o in zip(["center","log_dims","rot6d"], outs):
            print(f"      {name}: shape={o.shape}  dtype={o.dtype}")
    except ImportError:
        print("  ℹ  onnxruntime not installed — skipping verification")
        print("     pip install onnxruntime-gpu")
    except Exception as e:
        print(f"  ✗  ONNXRuntime error: {e}")

    # ── INT8 dynamic quantisation ─────────────────────────────────────────
    if not args.no_quantise:
        print("\nApplying INT8 dynamic quantisation …")
        try:
            from torch.quantization import quantize_dynamic
            model_q   = quantize_dynamic(
                model, {nn.Conv1d, nn.Linear}, dtype=torch.qint8)
            wrapper_q = OnnxWrapper(model_q)
            wrapper_q.eval()
            q_path = os.path.join(args.out_dir, "pointnetbbox_int8.onnx")
            torch.onnx.export(
                wrapper_q, dummy, q_path,
                opset_version = args.opset,
                input_names   = ["point_cloud"],
                output_names  = ["center","log_dims","rot6d"],
            )
            int8_mb = os.path.getsize(q_path) / 1e6
            print(f"  ✓  {q_path}  ({int8_mb:.1f} MB)")
            print(f"  Size reduction: {fp32_mb/int8_mb:.1f}×")

            # Latency comparison
            import time
            n_runs = 50
            with torch.no_grad():
                # FP32
                t0 = time.perf_counter()
                for _ in range(n_runs):
                    wrapper(dummy)
                fp32_ms = (time.perf_counter()-t0)/n_runs*1000

                # INT8
                t0 = time.perf_counter()
                for _ in range(n_runs):
                    wrapper_q(dummy)
                int8_ms = (time.perf_counter()-t0)/n_runs*1000

            print(f"  CPU latency (batch=1, N={num_points}):")
            print(f"    FP32: {fp32_ms:.1f} ms")
            print(f"    INT8: {int8_ms:.1f} ms  ({fp32_ms/int8_ms:.1f}× speedup)")

        except Exception as e:
            print(f"  Quantisation failed: {e}")

    # ── summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print("ONNX Export Summary")
    print(f"{'='*50}")
    for f in sorted(os.listdir(args.out_dir)):
        fp = os.path.join(args.out_dir, f)
        print(f"  {f:<35} {os.path.getsize(fp)/1e6:.1f} MB")
    print(f"\nDeploy with ONNX Runtime:")
    print(f"  sess = ort.InferenceSession('{onnx_path}')")
    print(f"  center, log_dims, rot6d = sess.run(None, {{'point_cloud': pts_np}})")
    print(f"\nDeploy with TensorRT:")
    print(f"  trtexec --onnx=pointnetbbox.onnx --fp16 --saveEngine=pointnetbbox.trt")


if __name__ == "__main__":
    main()