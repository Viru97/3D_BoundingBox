from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from sereact_3d_bbox.config import cfg
from sereact_3d_bbox.data.preprocessing import (
    DataValidationError,
    load_sample,
    preprocess_instance,
)


@dataclass(frozen=True)
class InstanceRecord:
    folder: Path
    scene_id: str
    instance_idx: int


class PointCloudInstanceDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        num_points: int = cfg.data.num_points,
        is_train: bool = True,
        scene_ids: list[str] | set[str] | None = None,
        seed: int = cfg.train.seed,
        strict: bool = False,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.num_points = num_points
        self.is_train = is_train
        self.seed = seed
        self.strict = strict
        self.scene_filter = set(scene_ids) if scene_ids is not None else None
        self.instances: list[InstanceRecord] = []
        self.skipped: list[str] = []
        self._scan()

    def _scan(self) -> None:
        folders = sorted(p for p in self.data_root.glob("*") if p.is_dir())
        for folder in folders:
            scene_id = folder.name
            if self.scene_filter is not None and scene_id not in self.scene_filter:
                continue
            try:
                sample = load_sample(folder, require_bboxes=True)
            except DataValidationError as exc:
                message = str(exc)
                self.skipped.append(message)
                if self.strict:
                    raise
                continue
            for instance_idx, mask in enumerate(sample.masks):
                if not bool(mask.any()):
                    self.skipped.append(f"{scene_id}: skipped empty mask {instance_idx}")
                    continue
                try:
                    preprocess_instance(
                        sample,
                        instance_idx,
                        num_points=self.num_points,
                        rng=np.random.default_rng(self.seed + len(self.instances)),
                        augment=False,
                        require_target=True,
                    )
                except DataValidationError as exc:
                    self.skipped.append(str(exc))
                    if self.strict:
                        raise
                    continue
                self.instances.append(InstanceRecord(folder, scene_id, instance_idx))

        if not self.instances:
            raise DataValidationError(
                f"No usable instances found under {self.data_root}. "
                "Check dataset path, required files, masks, and bbox3d.npy shapes."
            )

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, idx: int) -> dict[str, object]:
        record = self.instances[idx]
        sample = load_sample(record.folder, require_bboxes=True)
        rng_seed = None if self.is_train else self.seed + idx
        rng = np.random.default_rng(rng_seed)
        result = preprocess_instance(
            sample,
            record.instance_idx,
            num_points=self.num_points,
            rng=rng,
            augment=self.is_train,
            require_target=True,
        )
        assert result.target_corners is not None
        target = result.target_corners.astype(np.float32)
        return {
            "points": torch.from_numpy(result.features),
            "target": torch.from_numpy(target),
            "target_center": torch.from_numpy(target.mean(axis=0).astype(np.float32)),
            "target_dims": torch.from_numpy(_target_dims(target)),
            "anchor": torch.from_numpy(result.anchor.astype(np.float32)),
            "scene_id": result.scene_id,
            "instance_idx": result.instance_idx,
        }


def _target_dims(corners: np.ndarray) -> np.ndarray:
    edges = np.stack(
        [
            corners[1] - corners[0],
            corners[3] - corners[0],
            corners[4] - corners[0],
        ],
        axis=0,
    )
    dims = np.linalg.norm(edges, axis=1).astype(np.float32)
    return np.sort(dims)[::-1].copy()


def collate_fn(batch: list[dict[str, object]]) -> dict[str, object]:
    return {
        "points": torch.stack([item["points"] for item in batch]),
        "target": torch.stack([item["target"] for item in batch]),
        "target_center": torch.stack([item["target_center"] for item in batch]),
        "target_dims": torch.stack([item["target_dims"] for item in batch]),
        "anchor": torch.stack([item["anchor"] for item in batch]),
        "scene_id": [item["scene_id"] for item in batch],
        "instance_idx": torch.tensor([item["instance_idx"] for item in batch], dtype=torch.long),
    }
