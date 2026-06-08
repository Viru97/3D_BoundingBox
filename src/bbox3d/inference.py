from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from bbox3d.config import cfg
from bbox3d.data.preprocessing import DataValidationError, SampleData, load_sample, preprocess_instance
from bbox3d.models.dgcnn import build_model, rotation_6d_to_matrix


@dataclass(frozen=True)
class MaskPrediction:
    mask: np.ndarray
    score: float | None = None
    label: str | None = None


@dataclass(frozen=True)
class Prediction:
    corners: np.ndarray
    center: np.ndarray
    dims: np.ndarray
    rotation: np.ndarray
    anchor: np.ndarray
    score: float | None = None
    scene_id: str | None = None
    instance_idx: int | None = None


class MaskProvider(ABC):
    @abstractmethod
    def predict_masks(self, sample: SampleData) -> list[MaskPrediction]:
        raise NotImplementedError


class NpyMaskProvider(MaskProvider):
    def predict_masks(self, sample: SampleData) -> list[MaskPrediction]:
        return [MaskPrediction(mask=mask.astype(bool), score=None) for mask in sample.masks]


def checkpoint_model_kwargs(ckpt: dict[str, object]) -> dict[str, object]:
    args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    config = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    model_cfg = config.get("model", {}) if isinstance(config, dict) else {}
    in_channels = args.get("in_channels", model_cfg.get("in_channels", cfg.model.in_channels))
    version = args.get("model_version", model_cfg.get("version"))
    if version is None:
        version = "v1" if in_channels == 7 else cfg.model.version
    return {
        "version": version,
        "in_channels": in_channels,
        "k": args.get("k_neighbors", model_cfg.get("k_neighbors", cfg.model.k_neighbors)),
        "dropout": args.get("dropout", model_cfg.get("dropout", cfg.model.dropout)),
    }


def load_model(
    weights: str | Path,
    device: torch.device,
    allow_random_weights: bool = False,
    model_version: str | None = None,
    in_channels: int | None = None,
):
    weights = Path(weights)
    ckpt = None
    kwargs = {
        "version": model_version or cfg.model.version,
        "in_channels": in_channels or cfg.model.in_channels,
        "k": cfg.model.k_neighbors,
        "dropout": cfg.model.dropout,
    }
    if weights.exists():
        ckpt = torch.load(weights, map_location=device)
        kwargs.update(checkpoint_model_kwargs(ckpt))
    elif not allow_random_weights:
        raise FileNotFoundError(f"Checkpoint not found: {weights}")

    model = build_model(**kwargs).to(device)
    if ckpt is not None:
        state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        state = {key.replace("_orig_mod.", ""): value for key, value in state.items()}
        model.load_state_dict(state, strict=True)
    model.eval()
    return model, kwargs


def _predict_preprocessed(
    model,
    device: torch.device,
    features: np.ndarray,
    anchor: np.ndarray,
    score: float | None,
    scene_id: str,
    instance_idx: int,
) -> Prediction:
    inp = torch.from_numpy(features).unsqueeze(0).to(device)
    with torch.no_grad():
        center, log_dims, rot6d = model(inp)
        corners_rel = model.get_3d_box(center, log_dims, rot6d)
        rotation = rotation_6d_to_matrix(rot6d)

    anchor_np = anchor.astype(np.float32)
    center_abs = center[0].cpu().numpy() + anchor_np
    corners_abs = corners_rel[0].cpu().numpy() + anchor_np
    return Prediction(
        corners=corners_abs.astype(np.float32),
        center=center_abs.astype(np.float32),
        dims=torch.exp(log_dims[0]).cpu().numpy().astype(np.float32),
        rotation=rotation[0].cpu().numpy().astype(np.float32),
        anchor=anchor_np,
        score=score,
        scene_id=scene_id,
        instance_idx=instance_idx,
    )


def predict_sample(
    model,
    device: torch.device,
    sample_dir: str | Path,
    mask_provider: MaskProvider | None = None,
    num_points: int = cfg.data.num_points,
) -> list[Prediction]:
    sample = load_sample(sample_dir, require_bboxes=False)
    provider = mask_provider or NpyMaskProvider()
    mask_predictions = provider.predict_masks(sample)
    if not mask_predictions:
        return []

    predictions: list[Prediction] = []
    for idx, mask_prediction in enumerate(mask_predictions):
        proxy = SampleData(
            folder=sample.folder,
            scene_id=sample.scene_id,
            rgb=sample.rgb,
            point_cloud=sample.point_cloud,
            masks=np.asarray(mask_prediction.mask, dtype=bool)[None],
            bboxes=None,
        )
        try:
            result = preprocess_instance(
                proxy,
                0,
                num_points=num_points,
                rng=np.random.default_rng(42 + idx),
                augment=False,
                require_target=False,
            )
        except DataValidationError:
            continue
        predictions.append(
            _predict_preprocessed(
                model,
                device,
                result.features,
                result.anchor,
                mask_prediction.score,
                sample.scene_id,
                idx,
            )
        )
    return predictions
