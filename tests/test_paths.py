import json
from argparse import Namespace

from sereact_3d_bbox.paths import fill_missing_path_args, load_paths


def test_paths_file_fills_missing_args(tmp_path):
    paths_file = tmp_path / "paths.json"
    paths_file.write_text(
        json.dumps(
            {
                "data_root": "/data/root",
                "split_manifest": "/data/root/splits.json",
                "checkpoint": "/models/best.pth",
                "train_save_path": "/models/train.pth",
                "test_output_dir": "/tmp/test_output",
                "inference_output_dir": "/tmp/output",
            }
        ),
        encoding="utf-8",
    )

    args = Namespace(
        data_root=None,
        split_manifest=None,
        checkpoint=None,
        weights=None,
        save_path=None,
        vis_dir=None,
        out_dir=None,
        sample=None,
    )
    fill_missing_path_args(args, load_paths(paths_file))

    assert args.data_root == "/data/root"
    assert args.split_manifest == "/data/root/splits.json"
    assert args.checkpoint == "/models/best.pth"
    assert args.weights == "/models/best.pth"
    assert args.save_path == "/models/train.pth"
    assert args.vis_dir == "/tmp/test_output"
    assert args.out_dir == "/tmp/output"
