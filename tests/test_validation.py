from bbox3d.validation import validate_dataset

from .conftest import make_sample


def test_dataset_validation_reports_usable_and_skipped_instances(tmp_path):
    data_root = tmp_path / "dataset"
    make_sample(data_root, "valid")
    make_sample(data_root, "empty", empty_mask=True)

    report = validate_dataset(data_root, num_points=32)

    assert report.ready
    assert report.scenes_discovered == 2
    assert report.valid_scenes == 1
    assert report.invalid_scenes == 1
    assert report.total_instances == 2
    assert report.usable_instances == 1
    assert report.skipped_instances == 1
    assert report.errors
