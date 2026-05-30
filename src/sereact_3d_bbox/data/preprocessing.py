from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from sereact_3d_bbox.config import cfg


class DataValidationError(ValueError):
    """Raised when a dataset sample cannot be used safely."""


@dataclass(frozen=True)
class SampleData:
    folder: Path
    scene_id: str
    rgb: np.ndarray
    point_cloud: np.ndarray
    masks: np.ndarray
    bboxes: np.ndarray | None = None


@dataclass(frozen=True)
class PreprocessResult:
    features: np.ndarray
    anchor: np.ndarray
    target_corners: np.ndarray | None
    scene_id: str
    instance_idx: int
    floor_z: float
    object_points: int
    background_points: int


def _as_point_cloud_chw(pc: np.ndarray) -> np.ndarray:
    pc = np.asarray(pc, dtype=np.float32)
    if pc.ndim != 3:
        raise DataValidationError(f"pc.npy must be 3D, got shape {pc.shape}")
    if pc.shape[0] == 3:
        return pc
    if pc.shape[-1] == 3:
        return np.moveaxis(pc, -1, 0).astype(np.float32)
    raise DataValidationError(f"pc.npy must have 3 coordinate channels, got shape {pc.shape}")


def _as_masks_nhw(masks: np.ndarray, height: int, width: int) -> np.ndarray:
    masks = np.asarray(masks)
    if masks.ndim == 2:
        masks = masks[None]
    if masks.ndim != 3:
        raise DataValidationError(f"mask.npy must be (N,H,W) or (H,W), got {masks.shape}")
    if masks.shape[1:] != (height, width):
        raise DataValidationError(
            f"mask.npy spatial shape {masks.shape[1:]} does not match RGB {(height, width)}"
        )
    return masks.astype(bool)


def _as_bboxes_n83(bboxes: np.ndarray | None) -> np.ndarray | None:
    if bboxes is None:
        return None
    bboxes = np.asarray(bboxes, dtype=np.float32)
    if bboxes.ndim == 2 and bboxes.shape == (8, 3):
        bboxes = bboxes[None]
    if bboxes.ndim == 2 and bboxes.shape[1] == 24:
        bboxes = bboxes.reshape(-1, 8, 3)
    if bboxes.ndim != 3 or bboxes.shape[1:] != (8, 3):
        raise DataValidationError(f"bbox3d.npy must be (N,8,3) or (N,24), got {bboxes.shape}")
    if not np.isfinite(bboxes).all():
        raise DataValidationError("bbox3d.npy contains non-finite coordinates")
    return bboxes.astype(np.float32)


def load_sample(folder: str | os.PathLike, require_bboxes: bool = True) -> SampleData:
    folder = Path(folder)
    rgb_path = folder / "rgb.jpg"
    pc_path = folder / "pc.npy"
    mask_path = folder / "mask.npy"
    bbox_path = folder / "bbox3d.npy"

    missing = [p.name for p in (rgb_path, pc_path, mask_path) if not p.exists()]
    if require_bboxes and not bbox_path.exists():
        missing.append(bbox_path.name)
    if missing:
        raise DataValidationError(f"{folder}: missing required files: {', '.join(missing)}")

    img_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise DataValidationError(f"{rgb_path}: OpenCV could not read image")
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    pc = _as_point_cloud_chw(np.load(pc_path, allow_pickle=True))
    height, width = rgb.shape[:2]
    if pc.shape[1:] != (height, width):
        raise DataValidationError(
            f"pc.npy spatial shape {pc.shape[1:]} does not match RGB {(height, width)}"
        )

    masks = _as_masks_nhw(np.load(mask_path, allow_pickle=True), height, width)
    bboxes = None
    if bbox_path.exists():
        try:
            bboxes = _as_bboxes_n83(np.load(bbox_path, allow_pickle=True))
        except DataValidationError:
            if require_bboxes:
                raise
    if require_bboxes and bboxes is None:
        raise DataValidationError(f"{folder}: bbox3d.npy is required")
    if bboxes is not None and len(bboxes) != len(masks):
        if require_bboxes:
            raise DataValidationError(
                f"{folder}: mask count {len(masks)} does not match bbox count {len(bboxes)}"
            )
        bboxes = None

    return SampleData(folder=folder, scene_id=folder.name, rgb=rgb, point_cloud=pc, masks=masks, bboxes=bboxes)


def mad_filter(
    pc_pts: np.ndarray,
    rgb_pts: np.ndarray,
    threshold: float = cfg.data.mad_threshold,
    min_points: int = cfg.data.min_points_threshold,
) -> tuple[np.ndarray, np.ndarray]:
    if pc_pts.shape[1] < min_points:
        return pc_pts, rgb_pts
    med = np.median(pc_pts, axis=1, keepdims=True)
    mad = np.median(np.abs(pc_pts - med), axis=1, keepdims=True) + 1e-6
    inlier = np.all(np.abs(pc_pts - med) < threshold * mad, axis=0)
    if int(inlier.sum()) < min_points:
        return pc_pts, rgb_pts
    return pc_pts[:, inlier], rgb_pts[:, inlier]


def sample_aligned(
    points: np.ndarray,
    rgb: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    available = points.shape[1]
    if available == 0:
        return (
            np.zeros((3, count), dtype=np.float32),
            np.zeros((3, count), dtype=np.float32),
        )
    choice = rng.choice(available, count, replace=available < count)
    return points[:, choice], rgb[:, choice]


def estimate_floor_z(
    bg_points: np.ndarray,
    obj_points: np.ndarray,
    floor_quantile: float = cfg.data.floor_quantile,
) -> float:
    source = bg_points if bg_points.shape[1] > 0 else obj_points
    if source.shape[1] == 0:
        return 0.0
    return float(np.quantile(source[2], floor_quantile))


def _extract_points(sample: SampleData, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    obj_y, obj_x = np.where(mask)
    bg_y, bg_x = np.where(~mask)
    if len(obj_y) == 0:
        raise DataValidationError(f"{sample.scene_id}: empty instance mask")

    pc = sample.point_cloud
    rgb_chw = np.moveaxis(sample.rgb, -1, 0)
    obj_points = pc[:, obj_y, obj_x]
    obj_rgb = rgb_chw[:, obj_y, obj_x]
    bg_points = pc[:, bg_y, bg_x]
    bg_rgb = rgb_chw[:, bg_y, bg_x]
    return obj_points, obj_rgb, bg_points, bg_rgb


def _valid_depth(points: np.ndarray, rgb: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(points).all(axis=0)
    valid = finite & (points[2] > threshold)
    return points[:, valid], rgb[:, valid]


def _augment(
    features: np.ndarray,
    target: np.ndarray | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray | None]:
    theta = rng.uniform(0.0, 2.0 * np.pi)
    c, s = np.cos(theta), np.sin(theta)
    rot_z = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    features[:3] = rot_z @ features[:3]
    if target is not None:
        target = (rot_z @ target.T).T

    for axis in range(2):
        if rng.random() < 0.5:
            features[axis] *= -1.0
            if target is not None:
                target[:, axis] *= -1.0

    scale = rng.uniform(0.9, 1.1)
    features[:3] *= scale
    features[7:9] *= scale
    if target is not None:
        target *= scale

    z_shift = rng.uniform(-0.02, 0.02)
    features[2] += z_shift
    if target is not None:
        target[:, 2] += z_shift

    if rng.random() < 0.5:
        n_drop = int(rng.integers(1, max(2, features.shape[1] // 10)))
        drop = rng.choice(features.shape[1], n_drop, replace=False)
        features[:, drop] = 0.0

    features[:3] += rng.normal(0.0, 0.003, size=features[:3].shape).astype(np.float32)
    for channel in range(3, 6):
        alpha = rng.uniform(0.8, 1.2)
        beta = rng.uniform(-0.05, 0.05)
        features[channel] = np.clip(alpha * features[channel] + beta, 0.0, 1.0)
    return features, target


def preprocess_instance(
    sample: SampleData,
    instance_idx: int,
    num_points: int = cfg.data.num_points,
    rng: np.random.Generator | None = None,
    augment: bool = False,
    require_target: bool = True,
) -> PreprocessResult:
    if rng is None:
        rng = np.random.default_rng()
    if instance_idx < 0 or instance_idx >= len(sample.masks):
        raise DataValidationError(f"{sample.scene_id}: invalid instance index {instance_idx}")

    target = None
    if sample.bboxes is not None:
        target = sample.bboxes[instance_idx].astype(np.float32)
    elif require_target:
        raise DataValidationError(f"{sample.scene_id}: target bbox required for instance {instance_idx}")

    obj_points, obj_rgb, bg_points, bg_rgb = _extract_points(sample, sample.masks[instance_idx])
    obj_points, obj_rgb = _valid_depth(obj_points, obj_rgb, cfg.data.depth_threshold)
    bg_points, bg_rgb = _valid_depth(bg_points, bg_rgb, cfg.data.depth_threshold)
    obj_points, obj_rgb = mad_filter(obj_points, obj_rgb)

    if obj_points.shape[1] < cfg.data.min_points_threshold:
        raise DataValidationError(
            f"{sample.scene_id}: instance {instance_idx} has only {obj_points.shape[1]} valid object points"
        )

    anchor = np.median(obj_points, axis=1).astype(np.float32)
    floor_z = estimate_floor_z(bg_points, obj_points)

    n_obj = int(round(num_points * cfg.data.object_fraction))
    n_obj = min(max(n_obj, 1), num_points - 1)
    n_bg = num_points - n_obj

    obj_sample, obj_rgb_sample = sample_aligned(obj_points, obj_rgb, n_obj, rng)
    bg_sample, bg_rgb_sample = sample_aligned(bg_points, bg_rgb, n_bg, rng)

    pc_abs = np.concatenate([obj_sample, bg_sample], axis=1).astype(np.float32)
    rgb = np.concatenate([obj_rgb_sample, bg_rgb_sample], axis=1).astype(np.float32)
    mask_channel = np.concatenate(
        [np.ones((1, n_obj), dtype=np.float32), np.zeros((1, n_bg), dtype=np.float32)],
        axis=1,
    )

    centered_xyz = pc_abs - anchor[:, None]
    height_above_floor = (pc_abs[2:3] - floor_z).astype(np.float32)
    radial_distance = np.linalg.norm(centered_xyz, axis=0, keepdims=True).astype(np.float32)
    floor_contact = (
        (mask_channel < 0.5)
        & (np.abs(pc_abs[2:3] - floor_z) <= cfg.data.floor_contact_threshold)
    ).astype(np.float32)

    features = np.concatenate(
        [centered_xyz, rgb, mask_channel, height_above_floor, radial_distance, floor_contact],
        axis=0,
    ).astype(np.float32)

    target_offsets = target - anchor[None] if target is not None else None
    if augment:
        features, target_offsets = _augment(features, target_offsets, rng)

    return PreprocessResult(
        features=features.astype(np.float32),
        anchor=anchor,
        target_corners=None if target_offsets is None else target_offsets.astype(np.float32),
        scene_id=sample.scene_id,
        instance_idx=instance_idx,
        floor_z=floor_z,
        object_points=int(obj_points.shape[1]),
        background_points=int(bg_points.shape[1]),
    )
