"""
config.py — Centralized Configuration
FIX: Replaced @dataclass with plain classes using class-level attributes.
@dataclass creates *instance* attributes, so ModelConfig.in_channels would
raise AttributeError everywhere it was called.
"""
from pathlib import Path

DEFAULT_CHECKPOINT = "best_model.pth"
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_ONNX_DIR   = "onnx_export"

class ModelConfig:
    in_channels : int   = 7
    k_neighbors : int   = 20
    dropout     : float = 0.4

class DataConfig:
    num_points      : int   = 1024
    is_train_split  : float = 0.70
    val_split       : float = 0.10
    test_split      : float = 0.20
    depth_threshold : float = 0.01

class TrainConfig:
    batch_size    : int   = 32
    epochs        : int   = 100
    learning_rate : float = 5e-3
    weight_decay  : float = 5e-4
    warmup_epochs : int   = 5
    min_lr        : float = 1e-6
    grad_clip     : float = 1.0
    num_workers   : int   = 24