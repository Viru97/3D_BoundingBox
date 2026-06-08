from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class DataConfig:
    num_points: int = 1024
    object_fraction: float = 0.5
    mad_threshold: float = 5.0
    min_points_threshold: int = 10
    depth_threshold: float = 0.01
    floor_quantile: float = 0.20
    floor_contact_threshold: float = 0.02
    split_seed: int = 42
    train_ratio: float = 0.70
    val_ratio: float = 0.20
    test_ratio: float = 0.10
    split_manifest: str = "splits.json"

@dataclass
class ModelConfig:
    # XYZ + RGB + mask + height-above-floor + radial-distance + floor-contact.
    in_channels: int = 10
    k_neighbors: int = 20
    version: str = "v2"
    dropout: float = 0.30

@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    grad_clip: float = 1.0
    num_workers: int = 8
    seed: int = 42
    save_path: str = "best_model.pth"
    resume: str = ""
    chamfer_weight: float = 0.5
    corner_weight: float = 2.0
    center_weight: float = 2.0
    dimension_weight: float = 1.0
    rotation_weight: float = 0.08
    rotation_regularizer_weight: float = 0.005

@dataclass
class InferenceConfig:
    weights: str = "best_model.pth"
    out_dir: str = "output_visualizations"
    fail_on_missing_weights: bool = True

@dataclass
class EvalConfig:
    thresholds: tuple[float, ...] = (0.02, 0.05, 0.10, 0.20)
    failure_threshold: float = 0.10
    max_failures: int = 20

@dataclass
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

DEFAULT_DATA_ROOT = Path("dataset")
DEFAULT_CHECKPOINT = Path("best_model.pth")
DEFAULT_OUTPUT_DIR = Path("output")
DEFAULT_TEST_OUTPUT = Path("test_output")
DEFAULT_ONNX_DIR = Path("onnx_export")

cfg = AppConfig()
