"""
test.py  —  Held-out test evaluation for PointNetBBox
======================================================
Runs the best saved checkpoint against a dedicated test split (last 10%
of samples, never seen during training/validation) and reports:

  • Mean Corner Distance (MCD)  — primary metric, in metres
  • Median Corner Distance      — robust to outlier instances
  • Recall @ thresholds         — fraction of instances with MCD < τ
  • Per-object dimension error  — mean absolute error on predicted L/W/H

Usage:
    python test.py --data_root ~/Downloads/dl_challenge \
                   --checkpoint best_model.pth \
                   --vis_dir test_vis
"""

import os, sys, argparse, json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from scipy.optimize import linear_sum_assignment

# FIX: Removed collate_fn_test from import since it is defined below
from dataset import PointCloudInstanceDataset
from model   import PointNetBBox


# ── collate that also keeps raw targets for test metrics ─────────────────────
def collate_fn_test(batch):
    return {
        "points": torch.stack([b["points"] for b in batch]),
        "target": torch.stack([b["target"] for b in batch]),
    }


# ── metrics ───────────────────────────────────────────────────────────────────
def hungarian_mcd(pred, target):
    """
    pred, target : (8, 3) numpy
    Returns matched mean-corner-L2-distance in metres.
    """
    cost    = np.linalg.norm(pred[:, None] - target[None], axis=-1)  # (8,8)
    r, c    = linear_sum_assignment(cost)
    return float(np.sqrt(((pred[r] - target[c])**2).sum(-1)).mean())


def box_lwh(corners):
    """
    Estimate L, W, H from 8 corners by taking the 3 edge vectors from corner 0.
    corners : (8, 3)
    """
    v1 = corners[1] - corners[0]
    v2 = corners[3] - corners[0]
    v3 = corners[4] - corners[0]
    return sorted([np.linalg.norm(v1),
                   np.linalg.norm(v2),
                   np.linalg.norm(v3)], reverse=True)  # [L, W, H]


# ── visualisation ─────────────────────────────────────────────────────────────
EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
         (0,4),(1,5),(2,6),(3,7)]

def plot_boxes(ax, corners, color, label):
    for k, (i, j) in enumerate(EDGES):
        ax.plot([corners[i,0],corners[j,0]],
                [corners[i,1],corners[j,1]],
                [corners[i,2],corners[j,2]],
                color=color, lw=1.5, label=(label if k==0 else None))


def save_sample_fig(pred_corners, gt_corners, mcd, save_path):
    fig = plt.figure(figsize=(7, 6))
    ax  = fig.add_subplot(111, projection="3d")
    for b in gt_corners:
        plot_boxes(ax, b, "green", "GT")
    for b in pred_corners:
        plot_boxes(ax, b, "red", "Pred")
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels): seen.setdefault(l, h)
    ax.legend(seen.values(), seen.keys(), fontsize=8)
    ax.set_title(f"MCD = {mcd*100:.1f} cm")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


# ── main ──────────────────────────────────────────────────────────────────────
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── load checkpoint ────────────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location=device)
    saved_args = ckpt.get("args", {})
    num_points  = saved_args.get("num_points", 1024)

    model = PointNetBBox(in_channels=6, num_points=num_points).to(device)
    # Safely extract state_dict
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded: {args.checkpoint}  (epoch {ckpt.get('epoch','?')}, "
          f"best val median MCD={ckpt.get('best_mcd', float('nan')):.4f} m)")

    # ── test dataset: last 10% of samples (held-out from train/val) ────────
    full_ds = PointCloudInstanceDataset(args.data_root,
                                        num_points=num_points,
                                        is_train=False)
    n        = len(full_ds)
    test_idx = list(range(int(0.90 * n), n))          # same split as train.py
    test_ds  = Subset(full_ds, test_idx)
    loader   = DataLoader(test_ds, batch_size=args.batch_size,
                          shuffle=False, num_workers=4,
                          collate_fn=collate_fn_test)
    print(f"Test instances: {len(test_ds)}")

    if args.vis_dir:
        os.makedirs(args.vis_dir, exist_ok=True)

    # ── evaluation loop ────────────────────────────────────────────────────
    all_mcd     = []
    all_dim_err = []
    thresholds  = [0.02, 0.05, 0.10, 0.20]   # metres
    recalls     = {t: 0 for t in thresholds}
    vis_count   = 0

    with torch.no_grad():
        for batch in loader:
            pts = batch["points"].to(device)          # (B, 6, N)
            tgt = batch["target"].to(device).view(-1, 8, 3).float()  # (B,8,3)

            # FIX: Only unpack 3 values based on current model.py
            center, log_dims, rot6d = model(pts)
            pred = model.get_3d_box(center, log_dims, rot6d).float()  # (B,8,3)

            for b in range(pts.shape[0]):
                p_np = pred[b].cpu().numpy()    # (8,3) — relative to anchor
                t_np = tgt[b].cpu().numpy()

                mcd = hungarian_mcd(p_np, t_np)
                all_mcd.append(mcd)

                for τ in thresholds:
                    if mcd < τ:
                        recalls[τ] += 1

                # Dimension error (L/W/H)
                p_lwh = box_lwh(p_np)
                t_lwh = box_lwh(t_np)
                all_dim_err.append(np.abs(np.array(p_lwh) - np.array(t_lwh)).mean())

                # Visualise worst/best and random samples
                if args.vis_dir and vis_count < args.max_vis:
                    save_sample_fig([p_np], [t_np], mcd,
                        os.path.join(args.vis_dir,
                                     f"{vis_count:04d}_mcd{mcd*100:.1f}cm.png"))
                    vis_count += 1

    # ── report ────────────────────────────────────────────────────────────
    n_test = len(all_mcd)
    mean_mcd   = float(np.mean(all_mcd))
    median_mcd = float(np.median(all_mcd))
    std_mcd    = float(np.std(all_mcd))
    mean_dim   = float(np.mean(all_dim_err))

    print("\n" + "="*55)
    print(f"{'TEST RESULTS':^55}")
    print(f"{'Dataset':30s} {args.data_root}")
    print(f"{'Checkpoint':30s} {args.checkpoint}")
    print(f"{'Test instances':30s} {n_test}")
    print("─"*55)
    print(f"{'Mean MCD (m)':30s} {mean_mcd:.4f}")
    print(f"{'Median MCD (m)':30s} {median_mcd:.4f}")
    print(f"{'Std MCD (m)':30s} {std_mcd:.4f}")
    print(f"{'Mean MCD (cm)':30s} {mean_mcd*100:.2f}")
    print(f"{'Median MCD (cm)':30s} {median_mcd*100:.2f}")
    print("─"*55)
    for τ in thresholds:
        r = recalls[τ] / max(n_test, 1)
        print(f"{'Recall @ '+str(int(τ*100))+'cm':30s} {r:.3f}  ({recalls[τ]}/{n_test})")
    print("─"*55)
    print(f"{'Mean dim error (m)':30s} {mean_dim:.4f}")
    print("="*55)

    # ── MCD histogram ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Test Set Evaluation — PointNetBBox", fontweight="bold")

    ax = axes[0]
    ax.hist(np.array(all_mcd)*100, bins=40, color="#2196F3", edgecolor="white")
    ax.axvline(mean_mcd*100,   color="red",    lw=2, label=f"Mean {mean_mcd*100:.1f} cm")
    ax.axvline(median_mcd*100, color="orange", lw=2, label=f"Median {median_mcd*100:.1f} cm")
    for τ in [2, 5, 10]:
        ax.axvline(τ, color="gray", lw=1, ls="--", alpha=0.6)
    ax.set_xlabel("Mean Corner Distance (cm)")
    ax.set_ylabel("Instance count")
    ax.set_title("MCD Distribution")
    ax.legend()

    ax = axes[1]
    recall_vals = [recalls[t]/max(n_test,1) for t in thresholds]
    bars = ax.bar([f"{int(t*100)} cm" for t in thresholds],
                  recall_vals, color="#4CAF50", edgecolor="white")
    for bar, v in zip(bars, recall_vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f"{v:.2f}", ha="center", fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("MCD Threshold")
    ax.set_ylabel("Recall")
    ax.set_title("Recall @ Threshold")

    plt.tight_layout()
    plot_path = os.path.join(args.vis_dir or ".", "test_metrics.png")
    plt.savefig(plot_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\nMetric plot saved: {plot_path}")

    # ── JSON summary ──────────────────────────────────────────────────────
    summary = {
        "checkpoint":     args.checkpoint,
        "n_test":         n_test,
        "mean_mcd_m":     round(mean_mcd, 5),
        "median_mcd_m":   round(median_mcd, 5),
        "std_mcd_m":      round(std_mcd, 5),
        "mean_dim_err_m": round(mean_dim, 5),
        "recall": {f"{int(t*100)}cm": round(recalls[t]/max(n_test,1), 4)
                   for t in thresholds},
    }
    json_path = os.path.join(args.vis_dir or ".", "test_results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"JSON summary saved: {json_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",   required=True)
    p.add_argument("--checkpoint",  default="best_model.pth")
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--vis_dir",     default="test_output")
    p.add_argument("--max_vis",     type=int, default=20,
                   help="Max individual sample plots to save")
    main(p.parse_args())