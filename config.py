"""Configuration — fixed for DL challenge dataset."""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Config:
    # ---- Data ----
    data_root: str = "./dl_challenge"
    train_ratio: float = 0.8

    # ---- Classes ----
    classes: List[str] = field(default_factory=lambda: ["Object"])

    # ---- Input ----
    input_height: int = 480
    input_width: int = 640
    down_ratio: int = 4
    max_objects: int = 30
    in_channels: int = 6  # 3 RGB + 3 XYZ

    # ---- Model ----
    backbone: str = "resnet18"
    neck_channels: int = 256
    head_channels: int = 64

    # ---- Training ----
    device: str = "cuda"
    batch_size: int = 8
    num_workers: int = 4
    max_epochs: int = 200
    lr: float = 5e-4
    weight_decay: float = 1e-5
    warmup_epochs: int = 5
    lr_milestones: List[int] = field(default_factory=lambda: [100, 150, 180])
    lr_gamma: float = 0.1
    grad_clip: float = 35.0
    use_amp: bool = True

    # ---- Loss weights ----
    hm_weight: float = 1.0
    off2d_weight: float = 1.0
    center_weight: float = 5.0
    edge_weight: float = 10.0       # direct supervision on edge vectors
    corner_weight: float = 20.0     # reconstructed corner loss

    # ---- Inference ----
    topk: int = 50
    score_thresh: float = 0.1      # LOWERED from 0.3
    depth_range: Tuple[float, float] = (0.1, 3.0)

    # ---- Output ----
    output_dir: str = "./output"
    log_interval: int = 10
    save_interval: int = 5
    eval_start: int = 1  # start evaluation from epoch 1

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def output_height(self) -> int:
        return self.input_height // self.down_ratio

    @property
    def output_width(self) -> int:
        return self.input_width // self.down_ratio