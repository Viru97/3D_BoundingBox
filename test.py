import os, sys, argparse, json
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from scipy.optimize import linear_sum_assignment

from dataset import PointCloudInstanceDataset
from model import DGCNNBBox


def collate_fn_test(batch):
    return {
        "points": torch.stack([b["points"] for b in batch]),
        "target": torch.stack([b["target"] for b in batch]),
    }


def hungarian_mcd(pred, target):
    cost = np.linalg.norm(pred[:, None] - target[None], axis=-1)
    r, c = linear_sum_assignment(cost)
    return float(np.sqrt(((pred[r] - target[c]) ** 2).sum(-1)).mean())


def box_lwh(corners):
    v1 = corners[1] - corners[0]
    v2 = corners[3] - corners[0]
    v3 = corners[4] - corners[0]
    return sorted([np.linalg.norm(v1), np.linalg.norm(v2), np.linalg.norm(v3)], reverse=True)


EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def plot_boxes(ax, corners, color, label):
    for k, (i, j) in enumerate(EDGES):
        ax.plot([corners[i, 0], corners[j, 0]], [corners[i, 1], corners[j, 1]], [corners[i, 2], corners[j, 2]],
                color=color, lw=1.5, label=(label if k == 0 else None))


def save_sample_fig(pred_corners, gt_corners, mcd, save_path):
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    for b in gt_corners: plot_boxes(ax, b, "green", "GT")
    for b in pred_corners: plot_boxes(ax, b, "red", "Pred")
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels): seen.setdefault(l, h)
    ax.legend(seen.values(), seen.keys(), fontsize=8)
    ax.set_title(f"MCD = {mcd * 100:.1f} cm")
    ax.set_xlabel("X");
    ax.set_ylabel("Y");
    ax.set_zlabel("Z")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device)
    saved_args = ckpt.get("args", {})
    num_points = saved_args.get("num_points", 1024)

    model = DGCNNBBox(in_channels=args.in_channels).to(device)
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    print(
        f"Loaded: {args.checkpoint}  (epoch {ckpt.get('epoch', '?')}, best val median MCD={ckpt.get('best_mcd', float('nan')):.4f} m)")

    full_ds = PointCloudInstanceDataset(args.data_root, num_points=num_points, is_train=False)
    n = len(full_ds)
    test_idx = list(range(int(0.90 * n), n))
    test_ds = Subset(full_ds, test_idx)
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn_test)
    print(f"Test instances: {len(test_ds)}")

    if args.vis_dir: os.makedirs(args.vis_dir, exist_ok=True)

    all_mcd = []
    all_dim_err = []

    all_z_center_err = []
    all_z_height_err = []

    thresholds = [0.02, 0.05, 0.10, 0.20]
    recalls = {t: 0 for t in thresholds}
    sample_records = []

    with torch.no_grad():
        for batch in loader:
            pts = batch["points"].to(device)
            tgt = batch["target"].to(device).view(-1, 8, 3).float()

            center, log_dims, rot6d = model(pts)
            pred = model.get_3d_box(center, log_dims, rot6d).float()

            for b in range(pts.shape[0]):
                p_np = pred[b].cpu().numpy()
                t_np = tgt[b].cpu().numpy()

                mcd = hungarian_mcd(p_np, t_np)
                all_mcd.append(mcd)
                sample_records.append((mcd, p_np, t_np))

                for τ in thresholds:
                    if mcd < τ: recalls[τ] += 1

                p_lwh, t_lwh = box_lwh(p_np), box_lwh(t_np)
                all_dim_err.append(np.abs(np.array(p_lwh) - np.array(t_lwh)).mean())

                p_z_center = p_np[:, 2].mean()
                t_z_center = t_np[:, 2].mean()
                all_z_center_err.append(np.abs(p_z_center - t_z_center))

                p_z_height = p_np[:, 2].max() - p_np[:, 2].min()
                t_z_height = t_np[:, 2].max() - t_np[:, 2].min()
                all_z_height_err.append(np.abs(p_z_height - t_z_height))

    if args.vis_dir:
        worst_dir = os.path.join(args.vis_dir, "worst_10cm_failures")
        os.makedirs(worst_dir, exist_ok=True)
        sample_records.sort(key=lambda x: x[0], reverse=True)
        worst_cases = [rec for rec in sample_records if rec[0] >= 0.10]

        print(f"\nFound {len(worst_cases)} predictions that failed the 10cm recall threshold.")
        if len(worst_cases) > 0:
            print(f"Visualising the worst {min(len(worst_cases), args.max_vis)} cases...")
            for i, (mcd, p_np, t_np) in enumerate(worst_cases[:args.max_vis]):
                save_path = os.path.join(worst_dir, f"rank{i + 1:02d}_mcd{mcd * 100:.1f}cm.png")
                save_sample_fig([p_np], [t_np], mcd, save_path)
            print(f"Saved failure visualisations to: {worst_dir}/")

    n_test = len(all_mcd)
    mean_mcd = float(np.mean(all_mcd))
    median_mcd = float(np.median(all_mcd))
    std_mcd = float(np.std(all_mcd))
    mean_dim = float(np.mean(all_dim_err))

    mean_z_center = float(np.mean(all_z_center_err))
    mean_z_height = float(np.mean(all_z_height_err))

    print("\n" + "=" * 55)
    print(f"{'TEST RESULTS':^55}")
    print(f"{'Dataset':30s} {args.data_root}")
    print(f"{'Checkpoint':30s} {args.checkpoint}")
    print(f"{'Test instances':30s} {n_test}")
    print("─" * 55)
    print(f"{'Mean MCD (m)':30s} {mean_mcd:.4f}")
    print(f"{'Median MCD (m)':30s} {median_mcd:.4f}")
    print(f"{'Std MCD (m)':30s} {std_mcd:.4f}")
    print(f"{'Mean MCD (cm)':30s} {mean_mcd * 100:.2f}")
    print(f"{'Median MCD (cm)':30s} {median_mcd * 100:.2f}")
    print("─" * 55)
    for τ in thresholds:
        r = recalls[τ] / max(n_test, 1)
        print(f"{'Recall @ ' + str(int(τ * 100)) + 'cm':30s} {r:.3f}  ({recalls[τ]}/{n_test})")
    print("─" * 55)
    print(f"{'Mean global dim error (m)':30s} {mean_dim:.4f}")
    print(f"{'Mean Z-Center Error (m)':30s} {mean_z_center:.4f}  <-- Z Occlusion impact")
    print(f"{'Mean Z-Height Error (m)':30s} {mean_z_height:.4f}  <-- Z Occlusion impact")
    print("=" * 55)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Test Set Evaluation — DGCNNBBox", fontweight="bold")

    ax = axes[0]
    ax.hist(np.array(all_mcd) * 100, bins=40, color="#2196F3", edgecolor="white")
    ax.axvline(mean_mcd * 100, color="red", lw=2, label=f"Mean {mean_mcd * 100:.1f} cm")
    ax.axvline(median_mcd * 100, color="orange", lw=2, label=f"Median {median_mcd * 100:.1f} cm")
    for τ in [2, 5, 10]: ax.axvline(τ, color="gray", lw=1, ls="--", alpha=0.6)
    ax.set_xlabel("Mean Corner Distance (cm)");
    ax.set_ylabel("Instance count")
    ax.set_title("MCD Distribution");
    ax.legend()

    ax = axes[1]
    recall_vals = [recalls[t] / max(n_test, 1) for t in thresholds]
    bars = ax.bar([f"{int(t * 100)} cm" for t in thresholds], recall_vals, color="#4CAF50", edgecolor="white")
    for bar, v in zip(bars, recall_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{v:.2f}", ha="center", fontsize=10)
    ax.set_ylim(0, 1.1);
    ax.set_xlabel("MCD Threshold");
    ax.set_ylabel("Recall")
    ax.set_title("Recall @ Threshold")

    plt.tight_layout()
    plot_path = os.path.join(args.vis_dir or ".", "test_metrics.png")
    plt.savefig(plot_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\nMetric plot saved: {plot_path}")

    summary = {
        "checkpoint": args.checkpoint,
        "n_test": n_test,
        "mean_mcd_m": round(mean_mcd, 5),
        "median_mcd_m": round(median_mcd, 5),
        "std_mcd_m": round(std_mcd, 5),
        "mean_dim_err_m": round(mean_dim, 5),
        "mean_z_center_err_m": round(mean_z_center, 5),
        "mean_z_height_err_m": round(mean_z_height, 5),
        "recall": {f"{int(t * 100)}cm": round(recalls[t] / max(n_test, 1), 4) for t in thresholds},
    }
    json_path = os.path.join(args.vis_dir or ".", "test_results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"JSON summary saved: {json_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--checkpoint", default="best_model.pth")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--vis_dir", default="test_output")
    p.add_argument("--max_vis", type=int, default=20)
    p.add_argument("--in_channels", type=int, default=7)
    main(p.parse_args())