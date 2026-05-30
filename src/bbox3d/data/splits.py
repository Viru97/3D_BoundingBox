from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from bbox3d.config import cfg


def discover_scene_ids(data_root: str | Path) -> list[str]:
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def dataset_fingerprint(data_root: str | Path, scene_ids: list[str]) -> dict[str, object]:
    root = Path(data_root)
    files = ("rgb.jpg", "pc.npy", "mask.npy", "bbox3d.npy")
    stamp = []
    for scene_id in scene_ids:
        folder = root / scene_id
        for name in files:
            path = folder / name
            if path.exists():
                stat = path.stat()
                stamp.append((scene_id, name, stat.st_size, int(stat.st_mtime)))
    return {"scene_count": len(scene_ids), "files": stamp[:5000]}


def create_split_manifest(
    data_root: str | Path,
    manifest_path: str | Path,
    train_ratio: float = cfg.data.train_ratio,
    val_ratio: float = cfg.data.val_ratio,
    test_ratio: float = cfg.data.test_ratio,
    seed: int = cfg.data.split_seed,
    overwrite: bool = False,
) -> dict[str, object]:
    manifest_path = Path(manifest_path)
    if manifest_path.exists() and not overwrite:
        return load_split_manifest(manifest_path)

    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    scene_ids = discover_scene_ids(data_root)
    rng = np.random.default_rng(seed)
    shuffled = np.array(scene_ids, dtype=object)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    if n_train + n_val > n:
        n_val = max(0, n - n_train)

    train = sorted(shuffled[:n_train].tolist())
    val = sorted(shuffled[n_train : n_train + n_val].tolist())
    test = sorted(shuffled[n_train + n_val :].tolist())

    manifest = {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_root": str(Path(data_root)),
        "seed": seed,
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "splits": {"train": train, "val": val, "test": test},
        "dataset": dataset_fingerprint(data_root, scene_ids),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def load_split_manifest(manifest_path: str | Path) -> dict[str, object]:
    with Path(manifest_path).open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    splits = manifest.get("splits", {})
    all_ids = [scene for names in splits.values() for scene in names]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError(f"Split manifest has overlapping scene IDs: {manifest_path}")
    for key in ("train", "val", "test"):
        if key not in splits:
            raise ValueError(f"Split manifest missing '{key}' split: {manifest_path}")
    return manifest


def validate_manifest_for_data_root(manifest: dict[str, object], data_root: str | Path) -> None:
    scene_ids = set(discover_scene_ids(data_root))
    manifest_scene_ids = {
        scene_id
        for split_names in manifest.get("splits", {}).values()
        for scene_id in split_names
    }
    if not scene_ids:
        raise FileNotFoundError(f"No scene folders found under dataset root: {data_root}")
    if not manifest_scene_ids:
        raise ValueError("Split manifest contains no scene IDs")
    missing = manifest_scene_ids - scene_ids
    if missing:
        examples = ", ".join(sorted(list(missing))[:5])
        raise ValueError(
            "Split manifest does not match the selected data_root. "
            f"Missing scene IDs under {data_root}: {examples}. "
            "Point split_manifest at the dataset root you are training on, "
            "or rerun training with --rebuild_splits."
        )


def resolve_split_manifest(
    data_root: str | Path,
    manifest_path: str | Path | None = None,
    seed: int = cfg.data.split_seed,
    overwrite: bool = False,
) -> dict[str, object]:
    if manifest_path is None:
        manifest_path = Path(data_root) / cfg.data.split_manifest
    manifest = create_split_manifest(data_root, manifest_path, seed=seed, overwrite=overwrite)
    validate_manifest_for_data_root(manifest, data_root)
    return manifest


def split_scene_ids(manifest: dict[str, object], split: str) -> list[str]:
    splits = manifest["splits"]
    if split not in splits:
        raise KeyError(f"Unknown split '{split}', expected one of {sorted(splits)}")
    return list(splits[split])
