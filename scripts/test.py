import argparse
import json
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sereact_3d_bbox.config import cfg
from sereact_3d_bbox.data.dataset import PointCloudInstanceDataset, collate_fn
from sereact_3d_bbox.data.splits import resolve_split_manifest, split_scene_ids
from sereact_3d_bbox.inference import load_model
from sereact_3d_bbox.metrics import angular_error_degrees, mean_corner_distance, sorted_box_dimensions, summarize_mcd
from sereact_3d_bbox.models.loss import BBoxLoss
from sereact_3d_bbox.paths import fill_missing_path_args, load_paths


EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def save_box_png(points, pred, gt, mcd, path, title=None):
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    if points is not None:
        xyz = points[:3]
        is_object = points[6] > 0.5
        ax.scatter(
            xyz[0, ~is_object],
            xyz[1, ~is_object],
            xyz[2, ~is_object],
            c="#B7C1CC",
            s=2,
            alpha=0.18,
            label="Context points",
        )
        ax.scatter(
            xyz[0, is_object],
            xyz[1, is_object],
            xyz[2, is_object],
            c="#2D7DD2",
            s=4,
            alpha=0.45,
            label="Object points",
        )
    for corners, color, label in [(gt, "green", "GT"), (pred, "red", "Pred")]:
        for edge_idx, (i, j) in enumerate(EDGES):
            ax.plot(
                [corners[i, 0], corners[j, 0]],
                [corners[i, 1], corners[j, 1]],
                [corners[i, 2], corners[j, 2]],
                color=color,
                lw=1.5,
                label=label if edge_idx == 0 else None,
            )
    ax.set_title(title or f"MCD = {mcd * 100:.1f} cm")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.view_init(elev=24, azim=-56)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


def save_metrics_png(all_mcd, summary, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(np.asarray(all_mcd) * 100.0, bins=40, color="#2D7DD2", edgecolor="white")
    axes[0].axvline(summary["median_mcd_m"] * 100.0, color="orange", lw=2, label="median")
    axes[0].set_xlabel("MCD (cm)")
    axes[0].set_ylabel("Count")
    axes[0].legend()
    labels = list(summary["recall"].keys())
    values = [summary["recall"][label] for label in labels]
    axes[1].bar(labels, values, color="#3CA370", edgecolor="white")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Recall")
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()


def main(args):
    fill_missing_path_args(args, load_paths(args.paths_file))
    if not args.data_root:
        raise SystemExit("Set data_root in paths.local.json or pass --data_root.")
    args.checkpoint = args.checkpoint or cfg.inference.weights
    args.vis_dir = args.vis_dir or "test_output"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_kwargs = load_model(args.checkpoint, device, allow_random_weights=False)
    print(f"Device: {device}")
    print(f"Loaded: {args.checkpoint} ({model_kwargs})")

    manifest = resolve_split_manifest(args.data_root, args.split_manifest, seed=args.seed, overwrite=False)
    test_scenes = split_scene_ids(manifest, "test")
    test_ds = PointCloudInstanceDataset(
        args.data_root,
        num_points=args.num_points,
        is_train=False,
        scene_ids=test_scenes,
        seed=args.seed,
    )
    if test_ds.skipped:
        print(f"Skipped unusable test instances: {len(test_ds.skipped)}")
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    print(f"Test scenes={len(test_scenes)} instances={len(test_ds)}")

    criterion = BBoxLoss()
    out_dir = Path(args.vis_dir)
    failures_dir = out_dir / "worst_failures"
    failures_dir.mkdir(parents=True, exist_ok=True)
    for stale in failures_dir.glob("*.png"):
        stale.unlink()

    all_mcd, all_dim_err, all_z_err, all_ang_err = [], [], [], []
    losses = []
    latencies = []
    records = []

    with torch.no_grad():
        for batch in loader:
            pts = batch["points"].to(device)
            tgt = batch["target"].to(device).float()
            target_center = batch["target_center"].to(device).float()
            target_dims = batch["target_dims"].to(device).float()

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            center, log_dims, rot6d = model(pts)
            pred = model.get_3d_box(center, log_dims, rot6d).float()
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append(((time.perf_counter() - t0) * 1000.0) / max(pts.shape[0], 1))

            loss, _ = criterion(pred, tgt, center, log_dims, rot6d, target_center, target_dims)
            losses.append(float(loss.cpu()))
            batch_mcd = mean_corner_distance(pred, tgt)
            all_mcd.extend(batch_mcd)

            pred_dims = torch.sort(torch.exp(log_dims), dim=1, descending=True)[0].cpu().numpy()
            tgt_dims = sorted_box_dimensions(tgt).cpu().numpy()
            pred_np = pred.cpu().numpy()
            tgt_np = tgt.cpu().numpy()
            pts_np = batch["points"].cpu().numpy()
            for i, mcd in enumerate(batch_mcd):
                all_dim_err.append(float(np.abs(pred_dims[i] - tgt_dims[i]).mean()))
                all_z_err.append(float(abs(pred_np[i, :, 2].mean() - tgt_np[i, :, 2].mean())))
                all_ang_err.append(angular_error_degrees(pred_np[i], tgt_np[i]))
                records.append((mcd, pred_np[i], tgt_np[i], pts_np[i], batch["scene_id"][i], int(batch["instance_idx"][i])))

    records.sort(key=lambda item: item[0], reverse=True)
    for rank, (mcd, pred, target, points, scene_id, instance_idx) in enumerate(records[: args.max_vis], start=1):
        if mcd >= args.failure_threshold:
            save_box_png(
                points,
                pred,
                target,
                mcd,
                failures_dir / f"rank{rank:02d}_{scene_id}_{instance_idx}_mcd{mcd * 100:.1f}cm.png",
            )

    summary = summarize_mcd(all_mcd, tuple(args.thresholds))
    summary.update(
        {
            "checkpoint": args.checkpoint,
            "model": model_kwargs,
            "split_manifest": str(args.split_manifest or Path(args.data_root) / cfg.data.split_manifest),
            "n_test": len(all_mcd),
            "test_loss": round(float(np.mean(losses)), 6),
            "latency_ms_per_obj": round(float(np.mean(latencies)), 3),
            "mean_dim_err_m": round(float(np.mean(all_dim_err)), 6),
            "mean_z_center_err_m": round(float(np.mean(all_z_err)), 6),
            "mean_angular_err_deg": round(float(np.mean(all_ang_err)), 3),
        }
    )

    print("\nTEST RESULTS")
    print(f"  Mean MCD:     {summary['mean_mcd_m'] * 100:.2f} cm")
    print(f"  Median MCD:   {summary['median_mcd_m'] * 100:.2f} cm")
    print(f"  P90/P95 MCD:  {summary['p90_mcd_m'] * 100:.2f} / {summary['p95_mcd_m'] * 100:.2f} cm")
    print(f"  Dim error:    {summary['mean_dim_err_m'] * 100:.2f} cm")
    print(f"  Z error:      {summary['mean_z_center_err_m'] * 100:.2f} cm")
    print(f"  Angle error:  {summary['mean_angular_err_deg']:.2f} deg")
    for label, value in summary["recall"].items():
        print(f"  Recall @{label}: {value:.3f}")

    save_metrics_png(all_mcd, summary, out_dir / "test_metrics.png")

    if args.readme_gallery_dir:
        gallery_dir = Path(args.readme_gallery_dir)
        gallery_dir.mkdir(parents=True, exist_ok=True)
        for stale in gallery_dir.glob("best_sample_*.png"):
            stale.unlink()
        save_metrics_png(all_mcd, summary, gallery_dir / "test_metrics.png")
        best_records = sorted(records, key=lambda item: item[0])[: args.readme_gallery_count]
        for rank, (mcd, pred, target, points, scene_id, instance_idx) in enumerate(best_records, start=1):
            save_box_png(
                points,
                pred,
                target,
                mcd,
                gallery_dir / f"best_sample_{rank:02d}.png",
                title=f"Best sample {rank}: MCD = {mcd * 100:.1f} cm",
            )
        print(f"README gallery: {gallery_dir}")

    with (out_dir / "test_results.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"JSON: {out_dir / 'test_results.json'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--paths_file", default=None)
    p.add_argument("--data_root", default=None)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--split_manifest", default=None)
    p.add_argument("--batch_size", type=int, default=cfg.train.batch_size)
    p.add_argument("--num_points", type=int, default=cfg.data.num_points)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=cfg.train.seed)
    p.add_argument("--vis_dir", default=None)
    p.add_argument("--max_vis", type=int, default=cfg.eval.max_failures)
    p.add_argument("--failure_threshold", type=float, default=cfg.eval.failure_threshold)
    p.add_argument("--thresholds", type=float, nargs="+", default=list(cfg.eval.thresholds))
    p.add_argument("--readme_gallery_dir", default=None)
    p.add_argument("--readme_gallery_count", type=int, default=3)
    main(p.parse_args())
