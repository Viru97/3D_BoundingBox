"""
test.py — Quantitative evaluation on the held-out test split
============================================================
Usage:
    python scripts/test.py --data_root ~/Downloads/dl_challenge \
                           --checkpoint best_model.pth \
                           --out_dir test_output

Outputs:
  • test_metrics.png     — MCD distribution + Recall@τ bar chart
  • test_results.json    — machine-readable summary
  • Interactive HTML plots for worst-10 failure cases
"""

import os, json, argparse
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from torch.utils.data import DataLoader, Subset
from scipy.optimize import linear_sum_assignment

from sereact_bbox.model      import DGCNNBBox
from sereact_bbox.dataset    import PointCloudInstanceDataset
from sereact_bbox.validation import create_splits
from sereact_bbox.config     import DataConfig, ModelConfig, DEFAULT_CHECKPOINT, DEFAULT_OUTPUT_DIR

EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]


# ── metrics ───────────────────────────────────────────────────────────────────
def hungarian_mcd(pred, target):
    cost = np.linalg.norm(pred[:,None] - target[None], axis=-1)
    r, c = linear_sum_assignment(cost)
    return float(np.sqrt(((pred[r]-target[c])**2).sum(-1)).mean())

def box_lwh(corners):
    v1,v2,v3 = corners[1]-corners[0], corners[3]-corners[0], corners[4]-corners[0]
    return sorted([np.linalg.norm(v) for v in [v1,v2,v3]], reverse=True)


# ── plotly helper ─────────────────────────────────────────────────────────────
def add_box(fig, corners, color, name):
    xl, yl, zl = [], [], []
    for i, j in EDGES:
        xl += [corners[i,0], corners[j,0], None]
        yl += [corners[i,1], corners[j,1], None]
        zl += [corners[i,2], corners[j,2], None]
    fig.add_trace(go.Scatter3d(x=xl, y=yl, z=zl, mode='lines',
                               line=dict(color=color, width=4), name=name))

def save_failure_html(p_np, t_np, mcd, path):
    fig = go.Figure()
    add_box(fig, t_np, 'green', 'GT')
    add_box(fig, p_np, 'red',   'Pred')
    fig.update_layout(title=f"MCD = {mcd*100:.1f} cm",
                      scene=dict(aspectmode='data'))
    fig.write_html(path)


# ── main ──────────────────────────────────────────────────────────────────────
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt       = torch.load(args.checkpoint, map_location=device)
    model_state = {k.replace('_orig_mod.', ''): v
                   for k, v in ckpt.get("model", ckpt).items()}

    model = DGCNNBBox(in_channels=ModelConfig.in_channels).to(device)
    model.load_state_dict(model_state, strict=False)
    model.eval()
    print(f"Loaded: {args.checkpoint}  "
          f"(epoch={ckpt.get('epoch','?')}, "
          f"best_val_mcd={ckpt.get('best_mcd', float('nan')):.4f}m)")

    full_ds = PointCloudInstanceDataset(args.data_root,
                                        num_points=DataConfig.num_points,
                                        is_train=False)
    _, _, test_idx = create_splits(len(full_ds), DataConfig.is_train_split,
                                   DataConfig.val_split, DataConfig.test_split)
    test_ds = Subset(full_ds, test_idx)
    loader  = DataLoader(test_ds, batch_size=32, shuffle=False,
                         num_workers=4,
                         collate_fn=lambda b: {
                             "points": torch.stack([x["points"] for x in b]),
                             "target": torch.stack([x["target"] for x in b]),
                         })
    print(f"Test instances: {len(test_ds)}")

    os.makedirs(args.out_dir, exist_ok=True)
    failures_dir = os.path.join(args.out_dir, "worst_failures")
    os.makedirs(failures_dir, exist_ok=True)

    all_mcd, all_dim_err, all_z_err = [], [], []
    thresholds = [0.02, 0.05, 0.10, 0.20]
    recalls    = {t: 0 for t in thresholds}
    records    = []   # (mcd, p_np, t_np)

    with torch.no_grad():
        for batch in loader:
            pts = batch["points"].to(device)
            tgt = batch["target"].to(device).view(-1, 8, 3).float()

            center, log_dims, rot6d = model(pts)
            pred = model.get_3d_box(center, log_dims, rot6d).float()

            for b in range(pts.shape[0]):
                p_np = pred[b].cpu().numpy()
                t_np = tgt[b].cpu().numpy()
                mcd  = hungarian_mcd(p_np, t_np)
                all_mcd.append(mcd)
                records.append((mcd, p_np, t_np))

                for τ in thresholds:
                    if mcd < τ: recalls[τ] += 1

                all_dim_err.append(np.abs(np.array(box_lwh(p_np)) -
                                          np.array(box_lwh(t_np))).mean())
                all_z_err.append(abs(p_np[:,2].mean() - t_np[:,2].mean()))

    # Save worst failure HTML plots
    records.sort(key=lambda x: x[0], reverse=True)
    for i, (mcd, p, t) in enumerate(records[:args.max_vis]):
        if mcd >= 0.10:
            save_failure_html(p, t, mcd,
                os.path.join(failures_dir, f"rank{i+1:02d}_mcd{mcd*100:.1f}cm.html"))

    # ── report ────────────────────────────────────────────────────────────
    n         = len(all_mcd)
    mean_mcd  = float(np.mean(all_mcd))
    med_mcd   = float(np.median(all_mcd))
    std_mcd   = float(np.std(all_mcd))
    mean_dim  = float(np.mean(all_dim_err))
    mean_z    = float(np.mean(all_z_err))

    print("\n" + "="*55)
    print(f"{'TEST RESULTS':^55}")
    print(f"  Test instances:      {n}")
    print(f"  Mean MCD:            {mean_mcd*100:.2f} cm")
    print(f"  Median MCD:          {med_mcd*100:.2f} cm")
    print(f"  Std MCD:             {std_mcd*100:.2f} cm")
    print(f"  Mean dim error:      {mean_dim*100:.2f} cm")
    print(f"  Mean Z-centre err:   {mean_z*100:.2f} cm")
    print("─"*55)
    for τ in thresholds:
        r = recalls[τ]/max(n,1)
        print(f"  Recall @ {int(τ*100):2d}cm:       {r:.3f}  ({recalls[τ]}/{n})")
    print("="*55)

    # ── metric plots ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Test Set Evaluation — DGCNNBBox", fontweight="bold")

    ax = axes[0]
    ax.hist(np.array(all_mcd)*100, bins=40, color="#2196F3", edgecolor="white")
    ax.axvline(mean_mcd*100, color="red",    lw=2, label=f"Mean {mean_mcd*100:.1f} cm")
    ax.axvline(med_mcd*100,  color="orange", lw=2, label=f"Median {med_mcd*100:.1f} cm")
    for τ in [2,5,10]: ax.axvline(τ, color="gray", lw=1, ls="--", alpha=0.6)
    ax.set_xlabel("MCD (cm)"); ax.set_ylabel("Count")
    ax.set_title("MCD Distribution"); ax.legend()

    ax = axes[1]
    rv = [recalls[t]/max(n,1) for t in thresholds]
    bars = ax.bar([f"{int(t*100)} cm" for t in thresholds], rv,
                  color="#4CAF50", edgecolor="white")
    for bar, v in zip(bars, rv):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f"{v:.2f}", ha="center", fontsize=10)
    ax.set_ylim(0, 1.1); ax.set_xlabel("Threshold"); ax.set_ylabel("Recall")
    ax.set_title("Recall @ Threshold")

    plt.tight_layout()
    plot_path = os.path.join(args.out_dir, "test_metrics.png")
    plt.savefig(plot_path, dpi=130, bbox_inches="tight"); plt.close()
    print(f"\nMetric plot: {plot_path}")

    # ── JSON ──────────────────────────────────────────────────────────────
    summary = {
        "checkpoint": args.checkpoint, "n_test": n,
        "mean_mcd_m": round(mean_mcd, 5), "median_mcd_m": round(med_mcd, 5),
        "std_mcd_m": round(std_mcd, 5),   "mean_dim_err_m": round(mean_dim, 5),
        "mean_z_center_err_m": round(mean_z, 5),
        "recall": {f"{int(t*100)}cm": round(recalls[t]/max(n,1), 4)
                   for t in thresholds},
    }
    json_path = os.path.join(args.out_dir, "test_results.json")
    with open(json_path, "w") as f: json.dump(summary, f, indent=2)
    print(f"JSON summary: {json_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",  required=True)
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--out_dir",    default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--max_vis",    type=int, default=10)
    main(p.parse_args())