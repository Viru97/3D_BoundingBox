"""Loss functions — fixed: FP32 focal loss + edge→corner reconstruction loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from model import edges_to_corners


def _gather(feat_map, indices):
    """(B, C, H, W), (B, K) → (B, K, C)"""
    B, C, H, W = feat_map.shape
    feat = feat_map.view(B, C, -1).permute(0, 2, 1)
    idx  = indices.unsqueeze(2).expand(-1, -1, C).long()
    return feat.gather(1, idx)


def focal_loss(pred, gt, alpha=2.0, beta=4.0):
    """Modified focal loss — ALWAYS computed in FP32 to prevent NaN."""
    # Force FP32 regardless of AMP context
    pred = pred.float()
    gt   = gt.float()
    pred = pred.clamp(1e-4, 1 - 1e-4)

    pos = gt.eq(1).float()
    neg = gt.lt(1).float()
    neg_w = (1 - gt).pow(beta)

    pos_loss = -(1 - pred).pow(alpha) * torch.log(pred) * pos
    neg_loss = -pred.pow(alpha) * torch.log(1 - pred) * neg_w * neg

    n_pos = pos.sum().clamp(min=1)
    loss = (pos_loss.sum() + neg_loss.sum()) / n_pos

    # Safety check
    if torch.isnan(loss) or torch.isinf(loss):
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    return loss


def smooth_l1(pred, target, mask, beta=0.05):
    """Smooth L1 loss (better for small residuals than L1)."""
    diff = F.smooth_l1_loss(pred, target, reduction="none", beta=beta)
    diff = diff.mean(dim=-1)  # per-object mean
    return (diff * mask).sum() / mask.sum().clamp(min=1)


class CenterNet3DLoss(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

    def forward(self, output, targets):
        # ---- Heatmap (always FP32) ----
        hm_loss = focal_loss(output["heatmap"], targets["heatmap"])

        # ---- Gather at object locations ----
        idx  = targets["indices"]
        mask = targets["mask"].reshape(-1)

        off_p = _gather(output["offset_2d"],     idx).reshape(-1, 2)
        ctr_p = _gather(output["center_offset"], idx).reshape(-1, 3)
        edg_p = _gather(output["edges"],         idx).reshape(-1, 9)

        off_t = targets["offset_2d"].reshape(-1, 2)
        ctr_t = targets["center_offset"].reshape(-1, 3)
        edg_t = targets["gt_edges"].reshape(-1, 9)
        cor_t = targets["gt_corners"].reshape(-1, 24)

        # ---- Per-head losses ----
        off_loss = smooth_l1(off_p, off_t, mask)
        ctr_loss = smooth_l1(ctr_p, ctr_t, mask)

        # Edge vector loss (direct supervision)
        edg_loss = smooth_l1(edg_p, edg_t, mask)

        # Reconstructed corner loss (structural)
        corner_offsets_pred = edges_to_corners(edg_p)     # (BK, 8, 3)
        corner_offsets_pred = corner_offsets_pred.reshape(-1, 24)
        cor_loss = smooth_l1(corner_offsets_pred, cor_t, mask)

        # ---- Total ----
        c = self.cfg
        total = (c.hm_weight     * hm_loss
               + c.off2d_weight  * off_loss
               + c.center_weight * ctr_loss
               + c.edge_weight   * edg_loss
               + c.corner_weight * cor_loss)

        # NaN safety
        if torch.isnan(total):
            total = torch.tensor(0.0, device=total.device, requires_grad=True)

        stats = dict(total=total, hm=hm_loss, off=off_loss,
                     ctr=ctr_loss, edg=edg_loss, cor=cor_loss)
        return total, {k: v.detach() for k, v in stats.items()}