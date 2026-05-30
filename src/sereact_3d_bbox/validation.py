from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from sereact_3d_bbox.config import cfg
from sereact_3d_bbox.data.preprocessing import DataValidationError, load_sample, preprocess_instance


@dataclass(frozen=True)
class DatasetValidationReport:
    data_root: str
    scenes_discovered: int
    valid_scenes: int
    invalid_scenes: int
    total_instances: int
    usable_instances: int
    skipped_instances: int
    errors: list[str]

    @property
    def ready(self) -> bool:
        return self.usable_instances > 0

    def to_dict(self) -> dict[str, object]:
        return {"ready": self.ready, **asdict(self)}


def validate_dataset(
    data_root: str | Path,
    *,
    num_points: int = cfg.data.num_points,
    require_bboxes: bool = True,
    seed: int = cfg.train.seed,
) -> DatasetValidationReport:
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {root}")

    folders = sorted(path for path in root.iterdir() if path.is_dir())
    if not folders:
        raise DataValidationError(f"No scene folders found under dataset root: {root}")

    valid_scenes = 0
    total_instances = 0
    usable_instances = 0
    errors: list[str] = []

    for scene_offset, folder in enumerate(folders):
        try:
            sample = load_sample(folder, require_bboxes=require_bboxes)
        except DataValidationError as exc:
            errors.append(str(exc))
            continue

        scene_usable = 0
        total_instances += len(sample.masks)
        for instance_idx in range(len(sample.masks)):
            try:
                preprocess_instance(
                    sample,
                    instance_idx,
                    num_points=num_points,
                    rng=np.random.default_rng(seed + scene_offset + instance_idx),
                    augment=False,
                    require_target=require_bboxes,
                )
            except DataValidationError as exc:
                errors.append(str(exc))
                continue
            scene_usable += 1

        if scene_usable:
            valid_scenes += 1
            usable_instances += scene_usable
        elif not sample.masks.size:
            errors.append(f"{sample.scene_id}: sample contains no instance masks")

    return DatasetValidationReport(
        data_root=str(root),
        scenes_discovered=len(folders),
        valid_scenes=valid_scenes,
        invalid_scenes=len(folders) - valid_scenes,
        total_instances=total_instances,
        usable_instances=usable_instances,
        skipped_instances=total_instances - usable_instances,
        errors=errors,
    )
