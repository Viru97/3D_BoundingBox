"""
export_onnx.py  —  Export DGCNNBBox to ONNX + INT8 quantisation
================================================================
Produces:
  • onnx_export/pointnetbbox.onnx       — FP32, universal deployment
  • onnx_export/pointnetbbox_int8.onnx  — INT8 dynamic quant (~4× smaller)

Usage:
    python export_onnx.py --checkpoint best_model.pth --out_dir onnx_export

TensorRT deployment:
    trtexec --onnx=pointnetbbox.onnx --fp16 --saveEngine=pointnetbbox.trt

ONNX Runtime inference:
    sess = ort.InferenceSession("pointnetbbox.onnx")
    center, log_dims, rot6d = sess.run(None, {"point_cloud": pts_np})
    # pts_np shape: (1, 7, 1024)  dtype: float32
"""

import os, argparse, time
import torch
import torch.nn as nn
from model import DGCNNBBox


class OnnxWrapper(nn.Module):
    """Strips T_feat output if present; returns flat tuple for ONNX."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        # Support both 3-output (current) and 4-output (PointNet) models
        center, log_dims, rot6d = out[0], out[1], out[2]
        return center, log_dims, rot6d


def benchmark(model, dummy, n=100, label=""):
    model.eval()
    with torch.no_grad():
        for _ in range(10):   # warm-up
            model(dummy)
        t0 = time.perf_counter()
        for _ in range(n):
            model(dummy)
        ms = (time.perf_counter() - t0) / n * 1000
    print(f"  {label:<20} {ms:.2f} ms / sample")
    return ms


def main(args):
    device = torch.device("cpu")   # export on CPU for portability
    os.makedirs(args.out_dir, exist_ok=True)

    # ── load model ────────────────────────────────────────────────────────
    model = DGCNNBBox(in_channels=7)
    if os.path.exists(args.checkpoint):
        ckpt  = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(ckpt.get("model", ckpt))
        epoch = ckpt.get("epoch", "?")
        mcd   = ckpt.get("best_mcd", float("nan"))
        print(f"Loaded: {args.checkpoint}  (epoch={epoch}, best_val_mcd={mcd:.4f}m)")
    else:
        print("[WARN] Checkpoint not found — using random weights")
    model.eval()

    wrapper = OnnxWrapper(model)
    wrapper.eval()

    dummy    = torch.randn(1, 7, 1024)
    fp32_path = os.path.join(args.out_dir, "pointnetbbox.onnx")

    # ── FP32 ONNX export ─────────────────────────────────────────────────
    print(f"\nExporting FP32 ONNX (opset=18) …")
    torch.onnx.export(
        wrapper, dummy, fp32_path,
        opset_version       = 18,
        do_constant_folding = True,
        input_names         = ["point_cloud"],
        output_names        = ["center", "log_dims", "rot6d"],
        dynamic_axes        = {
            "point_cloud": {0: "batch"},
            "center":      {0: "batch"},
            "log_dims":    {0: "batch"},
            "rot6d":       {0: "batch"},
        },
    )
    fp32_mb = os.path.getsize(fp32_path) / 1e6
    print(f"  ✓  {fp32_path}  ({fp32_mb:.1f} MB)")

    # ── ONNX Runtime verification ─────────────────────────────────────────
    try:
        import onnxruntime as ort
        sess  = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
        outs  = sess.run(None, {"point_cloud": dummy.numpy()})
        print(f"  ✓  ONNXRuntime verification: "
              f"center{tuple(outs[0].shape)} "
              f"log_dims{tuple(outs[1].shape)} "
              f"rot6d{tuple(outs[2].shape)}")
    except ImportError:
        print("  ℹ  onnxruntime not installed — skipping verification")
        print("     pip install onnxruntime-gpu")

    # ── INT8 dynamic quantisation ─────────────────────────────────────────
    print("\nApplying INT8 dynamic quantisation …")
    try:
        from torch.quantization import quantize_dynamic
        model_q   = quantize_dynamic(model, {nn.Conv1d, nn.Conv2d, nn.Linear},
                                     dtype=torch.qint8)
        wrapper_q = OnnxWrapper(model_q)
        wrapper_q.eval()

        int8_path = os.path.join(args.out_dir, "pointnetbbox_int8.onnx")
        torch.onnx.export(
            wrapper_q, dummy, int8_path,
            opset_version = 18,
            input_names   = ["point_cloud"],
            output_names  = ["center", "log_dims", "rot6d"],
        )
        int8_mb = os.path.getsize(int8_path) / 1e6
        print(f"  ✓  {int8_path}  ({int8_mb:.1f} MB)")
        print(f"  Size reduction: {fp32_mb/int8_mb:.1f}×")

        # ── Latency benchmark ─────────────────────────────────────────────
        print("\nCPU latency benchmark (batch=1, N=1024, 100 runs):")
        fp32_ms = benchmark(wrapper,   dummy, label="FP32 PyTorch")
        int8_ms = benchmark(wrapper_q, dummy, label="INT8 PyTorch")
        print(f"  Speedup: {fp32_ms/int8_ms:.2f}×")

    except Exception as e:
        print(f"  INT8 quantisation failed: {e}")

    # ── summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print("Export Summary")
    print(f"{'='*50}")
    for f in sorted(os.listdir(args.out_dir)):
        fpath = os.path.join(args.out_dir, f)
        print(f"  {f:<40} {os.path.getsize(fpath)/1e6:.1f} MB")
    print(f"\nONNX Runtime deployment:")
    print(f"  import onnxruntime as ort")
    print(f"  sess = ort.InferenceSession('{fp32_path}')")
    print(f"  center, log_dims, rot6d = sess.run(None, {{'point_cloud': pts_np}})")
    print(f"\nTensorRT deployment:")
    print(f"  trtexec --onnx={fp32_path} --fp16 --saveEngine=model.trt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="best_model.pth")
    p.add_argument("--out_dir",    default="onnx_export")
    main(p.parse_args())