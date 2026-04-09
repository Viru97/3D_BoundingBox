"""Evaluation metrics — fixed: added debug output, robust IoU."""

import numpy as np


def box_volume(corners):
    """Volume from (8,3) corners."""
    e0 = corners[1] - corners[0]
    e1 = corners[3] - corners[0]
    e2 = corners[4] - corners[0]
    return abs(np.dot(e0, np.cross(e1, e2)))


def sample_in_box(corners, n, rng):
    e0 = corners[1] - corners[0]
    e1 = corners[3] - corners[0]
    e2 = corners[4] - corners[0]
    t = rng.random((n, 3)).astype(np.float32)
    return corners[0] + t[:, 0:1]*e0 + t[:, 1:2]*e1 + t[:, 2:3]*e2


def points_in_box(points, corners):
    e0 = corners[1] - corners[0]
    e1 = corners[3] - corners[0]
    e2 = corners[4] - corners[0]
    R = np.stack([e0, e1, e2], axis=1)
    try:
        d = points - corners[0]
        t = np.linalg.solve(R, d.T).T
    except np.linalg.LinAlgError:
        return np.zeros(len(points), dtype=bool)
    return np.all((t >= -1e-4) & (t <= 1 + 1e-4), axis=1)


def iou_3d(corners_a, corners_b, n_samples=5000):
    """3D IoU via bidirectional Monte Carlo."""
    vol_a = box_volume(corners_a)
    vol_b = box_volume(corners_b)
    if vol_a < 1e-12 or vol_b < 1e-12:
        return 0.0

    rng = np.random.default_rng(0)

    pts_a = sample_in_box(corners_a, n_samples, rng)
    in_b  = points_in_box(pts_a, corners_b).sum()

    pts_b = sample_in_box(corners_b, n_samples, rng)
    in_a  = points_in_box(pts_b, corners_a).sum()

    inter = 0.5 * (vol_a * in_b / n_samples + vol_b * in_a / n_samples)
    union = vol_a + vol_b - inter
    return float(np.clip(inter / max(union, 1e-12), 0, 1))


def corner_distance(corners_a, corners_b):
    """Mean L2 distance across 8 corners (in cm)."""
    return np.linalg.norm(corners_a - corners_b, axis=1).mean() * 100


def compute_ap(recall, precision, n_points=40):
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[0.0], precision, [0.0]])
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    thresholds = np.linspace(0, 1, n_points + 1)
    prec_interp = np.zeros_like(thresholds)
    for i, r in enumerate(thresholds):
        idx = np.where(mrec >= r)[0]
        if len(idx) > 0:
            prec_interp[i] = mpre[idx[0]]
    return prec_interp.mean()


def evaluate_ap(det_list, gt_list, iou_thresh=0.5, verbose=False):
    """AP with optional debug output."""
    all_scores, all_tp = [], []
    n_gt = 0
    iou_vals = []  # for debug

    for img_idx, (dets, gts) in enumerate(zip(det_list, gt_list)):
        n_gt += len(gts)
        if len(dets["scores"]) == 0:
            continue

        matched = [False] * len(gts)
        order = np.argsort(-dets["scores"])

        for idx in order:
            all_scores.append(dets["scores"][idx])
            best_iou, best_j = 0, -1
            for j in range(len(gts)):
                if matched[j]:
                    continue
                v = iou_3d(dets["corners"][idx], gts[j])
                if v > best_iou:
                    best_iou, best_j = v, j

            iou_vals.append(best_iou)

            if best_iou >= iou_thresh and best_j >= 0:
                matched[best_j] = True
                all_tp.append(1)
            else:
                all_tp.append(0)

    if verbose and iou_vals:
        iou_arr = np.array(iou_vals)
        print(f"    IoU stats: min={iou_arr.min():.4f} "
              f"median={np.median(iou_arr):.4f} "
              f"mean={iou_arr.mean():.4f} "
              f"max={iou_arr.max():.4f} "
              f"(>0.25: {(iou_arr>0.25).sum()}/{len(iou_arr)})")

    if n_gt == 0 or len(all_tp) == 0:
        return 0.0

    order = np.argsort(-np.array(all_scores))
    tp = np.array(all_tp)[order].cumsum()
    fp = (1 - np.array(all_tp)[order]).cumsum()
    recall    = tp / n_gt
    precision = tp / (tp + fp)
    return compute_ap(recall, precision)


def compute_all_metrics(det_list, gt_list, verbose=True):
    """Compute all metrics at once."""
    metrics = {}

    # AP at multiple IoU thresholds
    for iou_th in [0.25, 0.50]:
        ap = evaluate_ap(det_list, gt_list, iou_thresh=iou_th,
                         verbose=(verbose and iou_th == 0.25))
        metrics[f"AP@{iou_th}"] = ap

    # Corner & center errors (greedy matching at low IoU)
    corner_errs, center_errs = [], []
    for dets, gts in zip(det_list, gt_list):
        if len(dets["scores"]) == 0:
            continue
        order = np.argsort(-dets["scores"])
        matched = [False] * len(gts)
        for idx in order:
            best_iou, best_j = 0, -1
            for j in range(len(gts)):
                if matched[j]:
                    continue
                v = iou_3d(dets["corners"][idx], gts[j])
                if v > best_iou:
                    best_iou, best_j = v, j
            # Match at very low threshold for error analysis
            if best_iou > 0.05 and best_j >= 0:
                matched[best_j] = True
                corner_errs.append(corner_distance(
                    dets["corners"][idx], gts[best_j]))
                center_errs.append(
                    np.linalg.norm(dets["centers"][idx] -
                                   gts[best_j].mean(axis=0)) * 100)

    metrics["corner_err_cm"] = (float(np.mean(corner_errs))
                                 if corner_errs else float("nan"))
    metrics["center_err_cm"] = (float(np.mean(center_errs))
                                 if center_errs else float("nan"))
    metrics["n_matched"] = len(corner_errs)

    # Detection count stats
    n_det = sum(len(d["scores"]) for d in det_list)
    n_gt  = sum(len(g) for g in gt_list)
    metrics["n_det_total"] = n_det
    metrics["n_gt_total"]  = n_gt

    return metrics