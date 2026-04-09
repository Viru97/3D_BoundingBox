"""Post-processing — fixed BF16→numpy conversion."""

import numpy as np
import torch
import torch.nn.functional as F

from config import Config
from model import edges_to_corners


def heatmap_nms(hm, kernel=3):
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(hm, kernel, stride=1, padding=pad)
    return hm * (hmax == hm).float()


def topk_detections(hm, K=50):
    B, C, H, W = hm.shape
    hm_flat = hm.view(B, C, -1)
    topk_sc, topk_ind = torch.topk(hm_flat, K, dim=2)
    topk_ys = (topk_ind // W).float()
    topk_xs = (topk_ind %  W).float()
    topk_sc  = topk_sc.view(B, -1)
    topk_ind = topk_ind.view(B, -1)
    topk_ys  = topk_ys.view(B, -1)
    topk_xs  = topk_xs.view(B, -1)
    cls_ids = torch.arange(C, device=hm.device).view(1, C, 1).expand(
        B, C, K).reshape(B, -1)
    final_sc, sel = torch.topk(topk_sc, K, dim=1)
    return (final_sc,
            topk_ind.gather(1, sel),
            cls_ids.gather(1, sel),
            topk_ys.gather(1, sel),
            topk_xs.gather(1, sel))


def _gather_at(feat_map, indices):
    B, C, H, W = feat_map.shape
    feat = feat_map.view(B, C, -1).permute(0, 2, 1)
    idx  = indices.unsqueeze(2).expand(-1, -1, C).long()
    return feat.gather(1, idx)


def _to_numpy(t):
    """Safely convert any tensor (including BF16) to numpy float32."""
    return t.float().cpu().numpy()


@torch.no_grad()
def decode_predictions(output, pc_raw, cfg: Config):
    """Decode outputs → 3D bounding boxes."""
    hm = heatmap_nms(output["heatmap"])
    scores, sp_idx, cls_ids, ys, xs = topk_detections(hm, K=cfg.topk)

    off = _gather_at(output["offset_2d"],     sp_idx)
    ctr = _gather_at(output["center_offset"], sp_idx)
    edg = _gather_at(output["edges"],         sp_idx)

    img_x = (xs + off[..., 0]) * cfg.down_ratio
    img_y = (ys + off[..., 1]) * cfg.down_ratio

    EDGE_SCALE = 0.1  # meters
    corner_offsets = edges_to_corners(edg) * EDGE_SCALE

    B, K = scores.shape
    H, W = pc_raw.shape[2], pc_raw.shape[3]

    results = []
    for b in range(B):
        keep_mask = scores[b] > cfg.score_thresh
        n_keep = keep_mask.sum().item()

        if n_keep == 0:
            results.append(_empty_det())
            continue

        s   = _to_numpy(scores[b][keep_mask])
        ci  = _to_numpy(cls_ids[b][keep_mask]).astype(int)
        ix  = _to_numpy(img_x[b][keep_mask])
        iy  = _to_numpy(img_y[b][keep_mask])
        co  = _to_numpy(ctr[b][keep_mask])
        crn = _to_numpy(corner_offsets[b][keep_mask])

        pc_b = _to_numpy(pc_raw[b])

        centers_list, corners_list, valid = [], [], []
        for j in range(n_keep):
            px = int(np.clip(np.round(ix[j]), 0, W - 1))
            py = int(np.clip(np.round(iy[j]), 0, H - 1))

            pc_ref = _lookup_pc(pc_b, py, px, H, W)
            if pc_ref[2] < 0.01:
                continue

            center = pc_ref + co[j]
            corners = center + crn[j]

            if center[2] < cfg.depth_range[0] or center[2] > cfg.depth_range[1]:
                continue

            centers_list.append(center)
            corners_list.append(corners)
            valid.append(j)

        if valid:
            results.append(dict(
                scores  = s[valid],
                cls_ids = ci[valid],
                corners = np.array(corners_list),
                centers = np.array(centers_list),
            ))
        else:
            results.append(_empty_det())

    return results


def _lookup_pc(pc_b, py, px, H, W):
    val = pc_b[:, py, px].copy()
    if val[2] > 0.01:
        return val
    for w in range(1, 8):
        y0, y1 = max(0, py-w), min(H, py+w+1)
        x0, x1 = max(0, px-w), min(W, px+w+1)
        patch = pc_b[:, y0:y1, x0:x1]
        valid = patch[2] > 0.01
        if valid.any():
            return patch[:, valid].mean(axis=1)
    return np.zeros(3, dtype=np.float32)


def _empty_det():
    return dict(scores=np.zeros(0), cls_ids=np.zeros(0, dtype=int),
                corners=np.zeros((0, 8, 3)), centers=np.zeros((0, 3)))