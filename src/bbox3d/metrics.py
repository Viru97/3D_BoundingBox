from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch


UNIT_CORNERS = torch.tensor(
    [
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ],
    dtype=torch.float32,
)


@lru_cache(maxsize=1)
def cuboid_corner_permutations() -> torch.Tensor:
    corners = UNIT_CORNERS.numpy().astype(np.int32)
    lookup = {tuple(c): i for i, c in enumerate(corners)}
    perms: list[list[int]] = []

    axes = np.eye(3, dtype=np.int32)
    for order in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
        base = axes[list(order)]
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    mat = np.diag([sx, sy, sz]) @ base
                    if round(np.linalg.det(mat)) != 1:
                        continue
                    perm = [lookup[tuple(mat @ c)] for c in corners]
                    if perm not in perms:
                        perms.append(perm)
    return torch.tensor(perms, dtype=torch.long)


def chamfer_distance(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dist = torch.cdist(pred.float(), target.float())
    return dist.min(dim=2)[0].mean() + dist.min(dim=1)[0].mean()


def permutation_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    perms = cuboid_corner_permutations().to(target.device)
    target_perm = target[:, perms]
    d = (pred[:, None] - target_perm).pow(2).sum(dim=-1).sqrt().mean(dim=-1)
    return d.min(dim=1)[0]


def mean_corner_distance(pred: torch.Tensor, target: torch.Tensor) -> list[float]:
    return permutation_l2(pred.float(), target.float()).detach().cpu().tolist()


def box_dimensions(corners: torch.Tensor) -> torch.Tensor:
    edges = torch.stack(
        [
            corners[:, 1] - corners[:, 0],
            corners[:, 3] - corners[:, 0],
            corners[:, 4] - corners[:, 0],
        ],
        dim=1,
    )
    return edges.norm(dim=-1)


def sorted_box_dimensions(corners: torch.Tensor) -> torch.Tensor:
    return torch.sort(box_dimensions(corners), dim=1, descending=True)[0]


def angular_error_degrees(pred: np.ndarray, target: np.ndarray) -> float:
    pred_t = torch.from_numpy(pred[None].astype(np.float32))
    target_t = torch.from_numpy(target[None].astype(np.float32))
    perms = cuboid_corner_permutations()
    target_perm = target_t[:, perms]
    d = (pred_t[:, None] - target_perm).pow(2).sum(dim=-1).sqrt().mean(dim=-1)
    best = int(d.argmin(dim=1).item())
    aligned = target[perms[best].numpy()]

    pred_edges = [pred[1] - pred[0], pred[3] - pred[0], pred[4] - pred[0]]
    target_edges = [aligned[1] - aligned[0], aligned[3] - aligned[0], aligned[4] - aligned[0]]
    errors = []
    for p, t in zip(pred_edges, target_edges):
        p = p / (np.linalg.norm(p) + 1e-6)
        t = t / (np.linalg.norm(t) + 1e-6)
        errors.append(float(np.degrees(np.arccos(np.clip(np.dot(p, t), -1.0, 1.0)))))
    return float(np.mean(errors))


def summarize_mcd(values: list[float] | np.ndarray, thresholds: tuple[float, ...]) -> dict[str, object]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {
            "mean_mcd_m": float("nan"),
            "median_mcd_m": float("nan"),
            "std_mcd_m": float("nan"),
            "p90_mcd_m": float("nan"),
            "p95_mcd_m": float("nan"),
            "recall": {f"{int(t * 100)}cm": 0.0 for t in thresholds},
        }
    return {
        "mean_mcd_m": float(np.mean(arr)),
        "median_mcd_m": float(np.median(arr)),
        "std_mcd_m": float(np.std(arr)),
        "p90_mcd_m": float(np.percentile(arr, 90)),
        "p95_mcd_m": float(np.percentile(arr, 95)),
        "recall": {f"{int(t * 100)}cm": float(np.mean(arr < t)) for t in thresholds},
    }
