import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch

from bbox3d.config import cfg
from bbox3d.data.preprocessing import load_sample
from bbox3d.inference import NpyMaskProvider, load_model, predict_sample
from bbox3d.paths import fill_missing_path_args, load_paths


EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def ensure_writable_output_dir(out_dir):
    out_dir = Path(out_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        probe = out_dir / ".write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        raise SystemExit(
            f"Cannot write to inference output directory: {out_dir}\n"
            "Choose a writable path in paths.local.json as inference_output_dir "
            "or pass --out_dir /path/you/can/write."
        ) from exc
    return out_dir


def add_box(fig, corners, color, name):
    xs, ys, zs = [], [], []
    for i, j in EDGES:
        xs += [corners[i, 0], corners[j, 0], None]
        ys += [corners[i, 1], corners[j, 1], None]
        zs += [corners[i, 2], corners[j, 2], None]
    fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color=color, width=4), name=name))


def point_cloud_trace(sample):
    pc = sample.point_cloud.reshape(3, -1)
    pc = pc[:, np.isfinite(pc).all(axis=0) & (pc[2] > cfg.data.depth_threshold)]
    if pc.shape[1] == 0:
        return None
    med = np.median(pc, axis=1, keepdims=True)
    mad = np.median(np.abs(pc - med), axis=1, keepdims=True) + 1e-6
    inlier = np.all(np.abs(pc - med) < cfg.data.mad_threshold * mad, axis=0)
    if int(inlier.sum()) > 50:
        pc = pc[:, inlier]
    rng = np.random.default_rng(42)
    count = max(1, min(pc.shape[1], pc.shape[1] // 100))
    pc = pc[:, rng.choice(pc.shape[1], count, replace=False)]
    return go.Scatter3d(
        x=pc[0],
        y=pc[1],
        z=pc[2],
        mode="markers",
        marker=dict(size=1.5, color=pc[2], colorscale="Viridis", opacity=0.5),
        name="Point Cloud",
    )


def run_sample(model, device, sample_dir, out_dir, num_points):
    sample = load_sample(sample_dir, require_bboxes=False)
    predictions = predict_sample(model, device, sample_dir, NpyMaskProvider(), num_points=num_points)
    print(f"{sample.scene_id}: {len(predictions)} prediction(s)")

    fig = go.Figure()
    trace = point_cloud_trace(sample)
    if trace is not None:
        fig.add_trace(trace)
    for i, prediction in enumerate(predictions, start=1):
        add_box(fig, prediction.corners, "red", f"Pred {i}")
        dims = prediction.dims
        print(f"  Obj {i}: {dims[0]:.3f}m x {dims[1]:.3f}m x {dims[2]:.3f}m")
    if sample.bboxes is not None:
        for i, box in enumerate(sample.bboxes, start=1):
            add_box(fig, box, "green", f"GT {i}")

    fig.update_layout(
        title=f"3D Bounding Boxes: {sample.scene_id}",
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    out_path = Path(out_dir) / f"{sample.scene_id}_interactive.html"
    fig.write_html(out_path)
    print(f"  Saved: {out_path}")


def main(args):
    explicit_sample = args.sample is not None
    explicit_data_root = args.data_root is not None
    paths = load_paths(args.paths_file)
    fill_missing_path_args(args, paths)
    if explicit_sample and not explicit_data_root:
        args.data_root = None
    if args.sample and args.data_root:
        raise SystemExit("Use either --sample or --data_root, not both.")
    if not args.sample and not args.data_root:
        raise SystemExit("Set data_root or sample in paths.local.json, or pass --sample/--data_root.")
    args.weights = args.weights or cfg.inference.weights
    args.out_dir = args.out_dir or cfg.inference.out_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, kwargs = load_model(
        args.weights,
        device,
        allow_random_weights=args.allow_random_weights,
        model_version=args.model_version,
        in_channels=args.in_channels,
    )
    print(f"Loaded model: {kwargs}")
    args.out_dir = ensure_writable_output_dir(args.out_dir)

    if args.sample:
        samples = [Path(args.sample)]
    else:
        root = Path(args.data_root)
        if not root.is_dir():
            raise SystemExit(f"Dataset root is not a directory: {root}")
        samples = sorted(p for p in root.iterdir() if p.is_dir() and (p / "rgb.jpg").exists())
    if not samples:
        raise SystemExit("No samples found. Check --sample or data_root in paths.local.json.")

    for sample_dir in samples:
        run_sample(model, device, sample_dir, args.out_dir, args.num_points)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--paths_file", default=None)
    p.add_argument("--weights", default=None)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--num_points", type=int, default=cfg.data.num_points)
    p.add_argument("--model_version", choices=["v1", "v2"], default=None)
    p.add_argument("--in_channels", type=int, default=None)
    p.add_argument("--allow_random_weights", action="store_true")
    group = p.add_mutually_exclusive_group(required=False)
    group.add_argument("--sample", type=str)
    group.add_argument("--data_root", type=str)
    main(p.parse_args())
