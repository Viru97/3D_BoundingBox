import os
import cv2
import torch
import argparse
import numpy as np
import plotly.graph_objects as go
from sereact_bbox.model import DGCNNBBox
from sereact_bbox.config import DataConfig, ModelConfig, DEFAULT_OUTPUT_DIR, DEFAULT_CHECKPOINT

EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]


def add_plotly_box(fig, corners, color, name):
    x_lines, y_lines, z_lines = [], [], []
    for i, j in EDGES:
        x_lines.extend([corners[i, 0], corners[j, 0], None])
        y_lines.extend([corners[i, 1], corners[j, 1], None])
        z_lines.extend([corners[i, 2], corners[j, 2], None])

    fig.add_trace(
        go.Scatter3d(x=x_lines, y=y_lines, z=z_lines, mode='lines', line=dict(color=color, width=4), name=name))


def run_sample(model, device, sample_dir, out_dir, sample_idx, total, num_points=1024):
    img_bgr = cv2.imread(os.path.join(sample_dir, "rgb.jpg"))
    if img_bgr is None: return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pc = np.load(os.path.join(sample_dir, "pc.npy"), allow_pickle=True).astype(np.float32)
    masks = np.load(os.path.join(sample_dir, "mask.npy"), allow_pickle=True)

    folder_name = os.path.basename(sample_dir.rstrip("/"))
    print(f"\n[{sample_idx}/{total}] {folder_name} — {len(masks)} object(s)")

    predicted_boxes = []

    for i, mask in enumerate(masks):
        obj_y, obj_x = np.where(mask)
        bg_y, bg_x = np.where(~mask)
        if len(obj_y) == 0: continue

        pc_obj = pc[:, obj_y, obj_x]
        rgb_obj = img_rgb[obj_y, obj_x].astype(np.float32).T / 255.0

        pc_bg = pc[:, bg_y, bg_x]
        rgb_bg = img_rgb[bg_y, bg_x].astype(np.float32).T / 255.0

        # RGB already in [0,1] — no further normalisation (matches training)

        valid_obj = pc_obj[2] > 0.01
        pc_obj, rgb_obj = pc_obj[:, valid_obj], rgb_obj[:, valid_obj]

        valid_bg = pc_bg[2] > 0.01
        pc_bg, rgb_bg = pc_bg[:, valid_bg], rgb_bg[:, valid_bg]

        n_obj = num_points // 2
        n_bg = num_points - n_obj

        N_obj, N_bg = pc_obj.shape[1], pc_bg.shape[1]

        # Fixed random sampling
        if N_obj > 0:
            c_obj = np.random.choice(N_obj, n_obj, replace=(N_obj < n_obj))
            pc_obj, rgb_obj = pc_obj[:, c_obj], rgb_obj[:, c_obj]
        else:
            pc_obj, rgb_obj = np.zeros((3, n_obj), dtype=np.float32), np.zeros((3, n_obj), dtype=np.float32)

        if N_bg > 0:
            c_bg = np.random.choice(N_bg, n_bg, replace=(N_bg < n_bg))
            pc_bg, rgb_bg = pc_bg[:, c_bg], rgb_bg[:, c_bg]
        else:
            pc_bg, rgb_bg = np.zeros((3, n_bg), dtype=np.float32), np.zeros((3, n_bg), dtype=np.float32)

        mask_obj = np.ones((1, n_obj), dtype=np.float32)
        mask_bg = np.zeros((1, n_bg), dtype=np.float32)

        pc_comb = np.concatenate([pc_obj, pc_bg], axis=1)
        rgb_comb = np.concatenate([rgb_obj, rgb_bg], axis=1)
        mask_comb = np.concatenate([mask_obj, mask_bg], axis=1)

        anchor = np.median(pc_obj, axis=1) if N_obj > 0 else np.zeros(3, dtype=np.float32)
        pc_c = pc_comb - anchor.reshape(3, 1)

        features = np.concatenate([pc_c, rgb_comb, mask_comb], axis=0)
        inp = torch.from_numpy(features).unsqueeze(0).to(device)

        with torch.no_grad():
            center, log_dims, rot6d = model(inp)
            pred_corners = model.get_3d_box(center, log_dims, rot6d)

        corners_abs = pred_corners[0].cpu().numpy() + anchor
        predicted_boxes.append(corners_abs)

        # dims = torch.exp(log_dims[0]).cpu().numpy()
        # print(f"  Obj {i + 1}: {dims[0]:.3f}m × {dims[1]:.3f}m × {dims[2]:.3f}m")

    fig = go.Figure()

    pc_flat = pc.reshape(3, -1)
    valid = pc_flat[2] > 0.01
    pc_vis = pc_flat[:, valid].copy()

    n_sub = max(1, pc_vis.shape[1] // 100)
    sub = np.random.choice(pc_vis.shape[1], n_sub, replace=False)
    pc_sub = pc_vis[:, sub]

    fig.add_trace(go.Scatter3d(
        x=pc_sub[0], y=pc_sub[1], z=pc_sub[2],
        mode='markers', marker=dict(size=1.5, color=pc_sub[2], colorscale='Viridis', opacity=0.5), name='Point Cloud'
    ))

    for i, box in enumerate(predicted_boxes): add_plotly_box(fig, box, 'red', f'Pred Obj {i + 1}')

    gt_path = os.path.join(sample_dir, "bbox3d.npy")
    if os.path.exists(gt_path):
        gt_boxes = np.load(gt_path, allow_pickle=True)
        for i, box in enumerate(gt_boxes): add_plotly_box(fig, box, 'green', f'GT Obj {i + 1}')

    fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
                      title=f"Interactive 3D Bounding Boxes: {folder_name}", margin=dict(l=0, r=0, b=0, t=40))

    out_path = os.path.join(out_dir, f"{folder_name}_interactive.html")
    fig.write_html(out_path)
    print(f"  Saved Interactive 3D Plot: {out_path}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DGCNNBBox(in_channels=args.in_channels).to(device)

    if os.path.exists(args.weights):
        ckpt = torch.load(args.weights, map_location=device)
        model_state = ckpt.get("model", ckpt)
        model_state = {k.replace('_orig_mod.', ''): v for k, v in model_state.items()}
        model.load_state_dict(model_state, strict=False)

        # Use saved num_points from checkpoint if available
        saved_args = ckpt.get("args", {})
        num_points = args.num_points if args.num_points != 1024 else saved_args.get("num_points", 1024)
        print(f"Loaded DGCNN (7-Channel) weights successfully.")
    else:
        num_points = args.num_points
        print("[WARN] Weights not found!")

    model.eval()
    os.makedirs(args.out_dir, exist_ok=True)

    samples = [args.sample] if args.sample else sorted(
        p for p in (os.path.join(args.data_root, d) for d in os.listdir(args.data_root)) if
        os.path.isdir(p) and os.path.exists(os.path.join(p, "rgb.jpg")))

    for idx, sd in enumerate(samples, 1):
        run_sample(model, device, sd, args.out_dir, idx, len(samples), num_points=num_points)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default=str(DEFAULT_CHECKPOINT))
    p.add_argument("--out_dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--num_points", type=int, default=DataConfig.num_points)
    p.add_argument("--in_channels", type=int, default=ModelConfig.in_channels)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--sample", type=str)
    g.add_argument("--data_root", type=str)
    main(p.parse_args())