from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from bbox3d.config import cfg
from bbox3d.metrics import (
    chamfer_distance,
    cuboid_corner_permutations,
    mean_corner_distance,
    sorted_box_dimensions,
)
from bbox3d.models.dgcnn import rotation_6d_to_matrix


def cuboid_permutation_loss(pred: torch.Tensor, target: torch.Tensor, beta: float = 0.03) -> torch.Tensor:
    pred = pred.float()
    target = target.float()
    perms = cuboid_corner_permutations().to(target.device)
    target_perm = target[:, perms]
    err = F.smooth_l1_loss(
        pred[:, None].expand_as(target_perm),
        target_perm,
        beta=beta,
        reduction="none",
    ).mean(dim=(2, 3))
    return err.min(dim=1)[0].mean()


def rotation_raw_regularizer(rot6d: torch.Tensor) -> torch.Tensor:
    x = rot6d[:, :3]
    y = rot6d[:, 3:]
    x_norm = x.norm(dim=1)
    y_norm = y.norm(dim=1)
    dot = (F.normalize(x, dim=1, eps=1e-6) * F.normalize(y, dim=1, eps=1e-6)).sum(dim=1)
    return ((x_norm - 1.0).pow(2) + (y_norm - 1.0).pow(2) + dot.pow(2)).mean()


def target_pose_candidates(target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target = target.float()
    perms = cuboid_corner_permutations().to(target.device)
    target_perm = target[:, perms]
    edges = torch.stack(
        [
            target_perm[:, :, 1] - target_perm[:, :, 0],
            target_perm[:, :, 3] - target_perm[:, :, 0],
            target_perm[:, :, 4] - target_perm[:, :, 0],
        ],
        dim=2,
    )
    dims = edges.norm(dim=-1).clamp_min(1e-6)
    axes = edges / dims.unsqueeze(-1)
    rotations = axes.transpose(-1, -2).contiguous()
    return target_perm, dims, rotations


def matched_pose_losses(
    pred_corners: torch.Tensor,
    target_corners: torch.Tensor,
    pred_log_dims: torch.Tensor,
    pred_rot6d: torch.Tensor,
    beta: float = 0.03,
) -> dict[str, torch.Tensor]:
    pred_corners = pred_corners.float()
    target_corners = target_corners.float()
    pred_dims = torch.exp(pred_log_dims.float())
    pred_rot = rotation_6d_to_matrix(pred_rot6d.float())

    target_perm, target_dims, target_rot = target_pose_candidates(target_corners)

    corner_by_pose = F.smooth_l1_loss(
        pred_corners[:, None].expand_as(target_perm),
        target_perm,
        beta=beta,
        reduction="none",
    ).mean(dim=(2, 3))
    dim_by_pose = F.smooth_l1_loss(
        pred_dims[:, None].expand_as(target_dims),
        target_dims,
        beta=0.01,
        reduction="none",
    ).mean(dim=-1)
    axis_dot = (pred_rot[:, None] * target_rot).sum(dim=2).clamp(-1.0, 1.0)
    rotation_by_pose = (1.0 - axis_dot).mean(dim=-1)

    # Match all pose terms to the same cuboid symmetry.
    match_cost = corner_by_pose + dim_by_pose + 0.05 * rotation_by_pose
    best = match_cost.argmin(dim=1)
    rows = torch.arange(pred_corners.shape[0], device=pred_corners.device)
    return {
        "corner": corner_by_pose[rows, best].mean(),
        "dimension": dim_by_pose[rows, best].mean(),
        "rotation": rotation_by_pose[rows, best].mean(),
    }


class BBoxLoss(nn.Module):
    def __init__(
        self,
        chamfer_weight: float = cfg.train.chamfer_weight,
        corner_weight: float = cfg.train.corner_weight,
        center_weight: float = cfg.train.center_weight,
        dimension_weight: float = cfg.train.dimension_weight,
        rotation_weight: float = cfg.train.rotation_weight,
        rotation_regularizer_weight: float = cfg.train.rotation_regularizer_weight,
    ):
        super().__init__()
        self.chamfer_weight = chamfer_weight
        self.corner_weight = corner_weight
        self.center_weight = center_weight
        self.dimension_weight = dimension_weight
        self.rotation_weight = rotation_weight
        self.rotation_regularizer_weight = rotation_regularizer_weight

    def forward(
        self,
        pred_corners: torch.Tensor,
        target_corners: torch.Tensor,
        pred_center: torch.Tensor,
        pred_log_dims: torch.Tensor,
        pred_rot6d: torch.Tensor,
        target_center: torch.Tensor | None = None,
        target_dims: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        target_corners = target_corners.float()
        if target_center is None:
            target_center = target_corners.mean(dim=1)
        if target_dims is None:
            target_dims = sorted_box_dimensions(target_corners)

        pose_losses = matched_pose_losses(pred_corners, target_corners, pred_log_dims, pred_rot6d)
        pred_dims_sorted = torch.sort(torch.exp(pred_log_dims.float()), dim=1, descending=True)[0]
        sorted_dim_loss = F.smooth_l1_loss(pred_dims_sorted, target_dims.float(), beta=0.01)
        losses = {
            "chamfer": chamfer_distance(pred_corners, target_corners),
            "corner": pose_losses["corner"],
            "center": F.smooth_l1_loss(pred_center.float(), target_center.float(), beta=0.01),
            "dimension": pose_losses["dimension"] + 0.25 * sorted_dim_loss,
            "rotation": pose_losses["rotation"],
            "rotation_regularizer": rotation_raw_regularizer(pred_rot6d.float()),
        }
        total = (
            self.chamfer_weight * losses["chamfer"]
            + self.corner_weight * losses["corner"]
            + self.center_weight * losses["center"]
            + self.dimension_weight * losses["dimension"]
            + self.rotation_weight * losses["rotation"]
            + self.rotation_regularizer_weight * losses["rotation_regularizer"]
        )
        losses["total"] = total
        return total, losses


def compute_loss(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    center: torch.Tensor,
    log_dims: torch.Tensor | None = None,
    rot6d: torch.Tensor | None = None,
    target_center: torch.Tensor | None = None,
    target_dims: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if log_dims is None or rot6d is None:
        chamfer = chamfer_distance(pred.float(), tgt.float())
        corner = cuboid_permutation_loss(pred.float(), tgt.float())
        loss_center = F.smooth_l1_loss(center.float(), tgt.float().mean(dim=1), beta=0.01)
        return chamfer + corner + loss_center, chamfer, corner, loss_center
    criterion = BBoxLoss()
    total, losses = criterion(pred, tgt, center, log_dims, rot6d, target_center, target_dims)
    return total, losses["chamfer"], losses["corner"], losses["center"]


@torch.no_grad()
def mean_corner_dist(pred: torch.Tensor, target: torch.Tensor) -> list[float]:
    return mean_corner_distance(pred, target)
