from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_PATHS_FILE = "paths.local.json"


@dataclass
class ProjectPaths:
    data_root: str | None = None
    split_manifest: str | None = None
    checkpoint: str | None = None
    train_save_path: str | None = None
    test_output_dir: str | None = None
    inference_output_dir: str | None = None
    onnx_output_dir: str | None = None
    sample: str | None = None


def load_paths(paths_file: str | os.PathLike | None = None) -> ProjectPaths:
    path = Path(paths_file or os.environ.get("SEREACT_PATHS_FILE", DEFAULT_PATHS_FILE))
    if not path.exists():
        return ProjectPaths()
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a JSON object")
    allowed = ProjectPaths.__dataclass_fields__.keys()
    values: dict[str, Any] = {
        key: (value if value != "" else None)
        for key, value in raw.items()
        if key in allowed
    }
    return ProjectPaths(**values)


def fill_missing_path_args(args, paths: ProjectPaths) -> None:
    defaults = {
        "data_root": paths.data_root,
        "split_manifest": paths.split_manifest,
        "checkpoint": paths.checkpoint,
        "weights": paths.checkpoint,
        "save_path": paths.train_save_path or paths.checkpoint,
        "vis_dir": paths.test_output_dir,
        "out_dir": paths.inference_output_dir,
        "sample": paths.sample,
    }
    for name, value in defaults.items():
        if value and hasattr(args, name) and not getattr(args, name):
            setattr(args, name, value)


def fill_export_path_args(args, paths: ProjectPaths) -> None:
    if paths.checkpoint and not getattr(args, "checkpoint", None):
        args.checkpoint = paths.checkpoint
    if paths.onnx_output_dir and not getattr(args, "out_dir", None):
        args.out_dir = paths.onnx_output_dir
