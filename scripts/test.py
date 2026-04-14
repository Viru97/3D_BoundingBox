import os, json, argparse, time
import numpy as np
import torch
import matplotlib;

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from scipy.optimize import linear_sum_assignment

from sereact_3d_bbox.config import cfg
from sereact_3d_bbox.data.dataset import PointCloudInstanceDataset
from sereact_3d_bbox.models.dgcnn import DGCNNBBox


def collate_fn(batch):
    return {"points": torch.stack([b["points"] for b in batch]),
            "target": torch.stack([b["target"] for b in batch])}


def hungarian_mcd(pred, target):
    cost = np.linalg.norm(pred[:, None] - target[None], axis=-1)
    r, c = linear_sum_assignment(cost)
    return float(np.sqrt(((pred[r] - target[c]) ** 2).sum(-1)).mean()), r, c


def angular_error_degrees(pred, target, r, c):
    """Calculates the rotational error in degrees using the Hungarian matched corners."""
    aligned_tgt = target[c]
    aligned_pred = pred[r]

    # Extract primary axis vector (e.g., length-wise edge)
    v_p = aligned_pred[1] - aligned_pred[0]
    v_t = aligned_tgt[1] - aligned_tgt[0]

    v_p = v_p / (np.linalg.norm(v_p) + 1e-6)
    v_t = v_t / (np.linalg.norm(v_t) + 1e-6)

    cos_theta = np.clip(np.dot(v_p, v_t), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def box_lwh(corners):
    v1, v2, v3 = corners[1] - corners[0], corners[3] - corners[0], corners[4] - corners[0]
    return sorted([np.linalg.norm(v) for v in [v1, v2, v3]], reverse=True)


EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def save_failure_png(pred, gt, mcd, path):
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    for corners, color, label in [(gt, "green", "GT"), (pred, "red", "Pred")]:
        for k, (i, j) in enumerate(EDGES):
            ax.plot([corners[i, 0], corners[j, 0]], [corners[i, 1], corners[j, 1]],
                    [corners[i, 2], corners[j, 2]], color=color, lw=1.5,
                    label=(label if k == 0 else None))
    ax.set_title(f"MCD = {mcd * 100:.1f} cm")
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels): seen.setdefault(l, h)
    ax.legend(seen.values(), seen.keys(), fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=110, bbox_inches="tight");
    plt.close()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device)
    saved = ckpt.get("args", {})
    num_points = saved.get("num_points", 1024)
    in_channels = saved.get("in_channels", 7)

    model = DGCNNBBox(in_channels=in_channels).to(device)
    model.load_state_dict(ckpt.get("model", ckpt))
    model.eval()
    print(f"Loaded: {args.checkpoint}  ")

    full_ds = PointCloudInstanceDataset(args.data_root, num_points, is_train=False)
    n = len(full_ds)
    rng = np.random.default_rng(42)
    idx = rng.permutation(n)
    split = int(0.30 * n)
    n_val = int(0.10 * n)
    test_idx = idx[n_val:split].tolist()

    test_ds = Subset(full_ds, test_idx)
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, collate_fn=collate_fn)
    print(f"Test instances: {len(test_ds)}")

    os.makedirs(args.vis_dir, exist_ok=True)
    failures_dir = os.path.join(args.vis_dir, "worst_failures")
    os.makedirs(failures_dir, exist_ok=True)

    all_mcd, all_dim_err, all_z_err, all_ang_err = [], [], [], []
    latencies = []
    thresholds = [0.02, 0.05, 0.10, 0.20]
    recalls = {t: 0 for t in thresholds}
    records = []

    with torch.no_grad():
        for batch in loader:
            pts = batch["points"].to(device)
            tgt = batch["target"].to(device).view(-1, 8, 3).float()

            # Measure Latency
            t0 = time.perf_counter()
            center, log_dims, rot6d = model(pts)
            pred = model.get_3d_box(center, log_dims, rot6d).float()
            t1 = time.perf_counter()
            latencies.append(((t1 - t0) * 1000) / pts.shape[0])

            for b in range(pts.shape[0]):
                p_np, t_np = pred[b].cpu().numpy(), tgt[b].cpu().numpy()
                mcd, r_idx, c_idx = hungarian_mcd(p_np, t_np)

                all_mcd.append(mcd)
                records.append((mcd, p_np, t_np))

                for τ in thresholds:
                    if mcd < τ: recalls[τ] += 1

                all_dim_err.append(np.abs(np.array(box_lwh(p_np)) -
                                          np.array(box_lwh(t_np))).mean())
                all_z_err.append(abs(p_np[:, 2].mean() - t_np[:, 2].mean()))
                all_ang_err.append(angular_error_degrees(p_np, t_np, r_idx, c_idx))

    records.sort(key=lambda x: x[0], reverse=True)
    for i, (mcd, p, t) in enumerate(records[:args.max_vis]):
        if mcd >= 0.10:
            save_failure_png(p, t, mcd,
                             os.path.join(failures_dir, f"rank{i + 1:02d}_mcd{mcd * 100:.1f}cm.png"))

    n_test = len(all_mcd)
    mean_mcd = float(np.mean(all_mcd))
    med_mcd = float(np.median(all_mcd))
    std_mcd = float(np.std(all_mcd))
    mean_lat = float(np.mean(latencies))

    print("\n" + "=" * 55)
    print(f"{'TEST RESULTS':^55}")
    print("─" * 55)
    print(f"  Inference Latency:   {mean_lat:.2f} ms / object")
    print(f"  Mean MCD:            {mean_mcd * 100:.2f} cm")
    print(f"  Median MCD:          {med_mcd * 100:.2f} cm")
    print(f"  Std MCD:             {std_mcd * 100:.2f} cm")
    print(f"  Mean dim error:      {np.mean(all_dim_err) * 100:.2f} cm")
    print(f"  Mean Z-centre err:   {np.mean(all_z_err) * 100:.2f} cm")
    print(f"  Mean angular error:  {np.mean(all_ang_err):.2f}°")
    print("─" * 55)
    for τ in thresholds:
        r = recalls[τ] / max(n_test, 1)
        print(f"  Recall @ {int(τ * 100):2d} cm:      {r:.3f}  ({recalls[τ]}/{n_test})")
    print("=" * 55)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Test Set Evaluation — DGCNNBBox", fontweight="bold")
    ax = axes[0]
    ax.hist(np.array(all_mcd) * 100, bins=40, color="#2196F3", edgecolor="white")
    ax.axvline(mean_mcd * 100, color="red", lw=2, label=f"Mean {mean_mcd * 100:.1f} cm")
    ax.axvline(med_mcd * 100, color="orange", lw=2, label=f"Median {med_mcd * 100:.1f} cm")
    for τ in [2, 5, 10]: ax.axvline(τ, color="gray", lw=1, ls="--", alpha=0.5)
    ax.set_xlabel("MCD (cm)");
    ax.set_ylabel("Count");
    ax.legend()
    ax.set_title("MCD Distribution")
    ax = axes[1]
    rv = [recalls[t] / max(n_test, 1) for t in thresholds]
    bars = ax.bar([f"{int(t * 100)} cm" for t in thresholds], rv,
                  color="#4CAF50", edgecolor="white")
    for bar, v in zip(bars, rv):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.2f}", ha="center", fontsize=10)
    ax.set_ylim(0, 1.1);
    ax.set_title("Recall @ Threshold")
    plt.tight_layout()
    plot_path = os.path.join(args.vis_dir, "test_metrics.png")
    plt.savefig(plot_path, dpi=130, bbox_inches="tight");
    plt.close()
    print(f"\nPlot: {plot_path}")

    summary = {"checkpoint": args.checkpoint, "n_test": n_test,
               "latency_ms_per_obj": round(mean_lat, 2),
               "mean_mcd_m": round(mean_mcd, 5), "median_mcd_m": round(med_mcd, 5),
               "std_mcd_m": round(std_mcd, 5),
               "mean_dim_err_m": round(float(np.mean(all_dim_err)), 5),
               "mean_z_center_err_m": round(float(np.mean(all_z_err)), 5),
               "mean_angular_err_deg": round(float(np.mean(all_ang_err)), 2),
               "recall": {f"{int(t * 100)}cm": round(recalls[t] / max(n_test, 1), 4)
                          for t in thresholds}}
    json_path = os.path.join(args.vis_dir, "test_results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--checkpoint", default=cfg.inference.weights)
    p.add_argument("--batch_size", type=int, default=cfg.train.batch_size)
    p.add_argument("--vis_dir", default="test_output")
    p.add_argument("--max_vis", type=int, default=20)
    main(p.parse_args())