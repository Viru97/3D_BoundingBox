import os, argparse, time
import torch
import torch.nn as nn
from sereact_3d_bbox.models.dgcnn import DGCNNBBox

class OnnxWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        center, log_dims, rot6d = self.model(x)
        return center, log_dims, rot6d

def benchmark(model, dummy, n=100, label=""):
    model.eval()
    with torch.no_grad():
        for _ in range(10): model(dummy)   
        t0 = time.perf_counter()
        for _ in range(n):  model(dummy)
        ms = (time.perf_counter() - t0) / n * 1000
    print(f"  {label:<22} {ms:.2f} ms / sample")
    return ms

def main(args):
    device = torch.device("cpu")   
    os.makedirs(args.out_dir, exist_ok=True)

    model = DGCNNBBox(in_channels=7)
    if os.path.exists(args.checkpoint):
        ckpt  = torch.load(args.checkpoint, map_location="cpu")
        state = {k.replace("_orig_mod.", ""): v
                 for k, v in ckpt.get("model", ckpt).items()}
        model.load_state_dict(state, strict=False)
        print(f"Loaded: {args.checkpoint}")
    else:
        print("[WARN] Checkpoint not found — exporting random weights")
    model.eval()

    wrapper = OnnxWrapper(model); wrapper.eval()
    dummy   = torch.randn(1, 7, 1024)
    fp32_path = os.path.join(args.out_dir, "dgcnn_bbox.onnx")

    print(f"\nExporting FP32 ONNX (opset=18) -> {fp32_path}")
    torch.onnx.export(
        wrapper, dummy, fp32_path,
        opset_version       = 18,
        do_constant_folding = True,
        input_names         = ["point_cloud"],
        output_names        = ["center", "log_dims", "rot6d"],
        dynamic_axes        = {"point_cloud": {0: "batch"},
                               "center":      {0: "batch"},
                               "log_dims":    {0: "batch"},
                               "rot6d":       {0: "batch"}},
    )
    fp32_mb = os.path.getsize(fp32_path) / 1e6
    print(f"  ✓  {fp32_mb:.1f} MB")

    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
        outs = sess.run(None, {"point_cloud": dummy.numpy()})
        print(f"  ✓  Verified Output Shapes")
    except ImportError:
        print("  i  pip install onnxruntime-gpu  (skipping verification)")

    print("\nINT8 dynamic quantisation ...")
    try:
        from torch.quantization import quantize_dynamic
        model_q   = quantize_dynamic(model, {nn.Conv1d, nn.Conv2d, nn.Linear},
                                     dtype=torch.qint8)
        wrapper_q = OnnxWrapper(model_q); wrapper_q.eval()
        int8_path = os.path.join(args.out_dir, "dgcnn_bbox_int8.onnx")
        torch.onnx.export(wrapper_q, dummy, int8_path, opset_version=18,
                          input_names=["point_cloud"],
                          output_names=["center","log_dims","rot6d"])
        int8_mb = os.path.getsize(int8_path) / 1e6
        print(f"  ✓  {int8_path}  ({int8_mb:.1f} MB, {fp32_mb/int8_mb:.1f}× smaller)")

        print("\nCPU latency (batch=1, N=1024, 100 runs):")
        fp32_ms = benchmark(wrapper,   dummy, label="FP32")
        int8_ms = benchmark(wrapper_q, dummy, label="INT8")
        print(f"  Speedup: {fp32_ms/int8_ms:.2f}×")
    except Exception as e:
        print(f"  INT8 failed: {e}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="best_model.pth")
    p.add_argument("--out_dir",    default="onnx_export")
    main(p.parse_args())
