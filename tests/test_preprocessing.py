import numpy as np
import pytest

from bbox3d.data.dataset import PointCloudInstanceDataset
from bbox3d.data.preprocessing import DataValidationError, load_sample, preprocess_instance, sample_aligned

from .conftest import make_sample


def test_sample_aligned_keeps_point_rgb_pairs():
    points = np.vstack([np.arange(20), np.arange(20) + 100, np.arange(20) + 200]).astype(np.float32)
    rgb = np.vstack([np.arange(20) + 1, np.arange(20) + 2, np.arange(20) + 3]).astype(np.float32)
    sampled_points, sampled_rgb = sample_aligned(points, rgb, 12, np.random.default_rng(0))
    assert np.all(sampled_rgb[0] == sampled_points[0] + 1)
    assert np.all(sampled_rgb[1] == sampled_points[0] + 2)


def test_preprocess_returns_shared_feature_contract(tmp_path):
    folder = make_sample(tmp_path, "scene")
    sample = load_sample(folder)
    result = preprocess_instance(sample, 0, num_points=64, rng=np.random.default_rng(1), augment=False)
    assert result.features.shape == (10, 64)
    assert result.target_corners.shape == (8, 3)
    assert np.isfinite(result.features).all()
    assert result.object_points >= 10


def test_empty_masks_are_rejected(tmp_path):
    data_root = tmp_path / "dataset"
    make_sample(data_root, "empty", empty_mask=True)
    with pytest.raises(DataValidationError):
        PointCloudInstanceDataset(data_root, num_points=64)


def test_low_point_instances_are_skipped_at_scan(tmp_path):
    data_root = tmp_path / "dataset"
    folder = make_sample(data_root, "mixed", objects=2)
    masks = np.load(folder / "mask.npy")
    masks[0] = False
    masks[0, 1:4, 1:4] = True
    np.save(folder / "mask.npy", masks)

    ds = PointCloudInstanceDataset(data_root, num_points=64)
    assert len(ds) == 1
    assert any("only 9 valid object points" in message for message in ds.skipped)
