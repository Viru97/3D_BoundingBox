import argparse
import json
from pathlib import Path

from sereact_3d_bbox.config import cfg
from sereact_3d_bbox.paths import fill_missing_path_args, load_paths
from sereact_3d_bbox.validation import validate_dataset


def main(args):
    fill_missing_path_args(args, load_paths(args.paths_file))
    if not args.data_root:
        raise SystemExit("Set data_root in paths.local.json or pass --data_root.")

    report = validate_dataset(
        args.data_root,
        num_points=args.num_points,
        require_bboxes=not args.inference_only,
        seed=args.seed,
    )
    payload = report.to_dict()
    print("DATASET PREFLIGHT")
    print(f"  Root:              {report.data_root}")
    print(f"  Scenes:            {report.valid_scenes}/{report.scenes_discovered} usable")
    print(f"  Instances:         {report.usable_instances}/{report.total_instances} usable")
    print(f"  Skipped instances: {report.skipped_instances}")
    if report.errors:
        print(f"  Validation errors: {len(report.errors)}")
        for error in report.errors[: args.max_errors]:
            print(f"    - {error}")
        if len(report.errors) > args.max_errors:
            print(f"    ... {len(report.errors) - args.max_errors} more")

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"JSON: {out_path}")

    if not report.ready:
        raise SystemExit("Dataset preflight failed: no usable instances found.")
    if args.strict and report.errors:
        raise SystemExit("Dataset preflight failed in strict mode: validation errors found.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Validate dataset files and preprocess every mask before training.")
    p.add_argument("--paths_file", default=None)
    p.add_argument("--data_root", default=None)
    p.add_argument("--num_points", type=int, default=cfg.data.num_points)
    p.add_argument("--seed", type=int, default=cfg.train.seed)
    p.add_argument("--inference_only", action="store_true", help="Allow samples without bbox3d.npy targets.")
    p.add_argument("--strict", action="store_true", help="Exit non-zero if any scene or instance is skipped.")
    p.add_argument("--max_errors", type=int, default=20)
    p.add_argument("--json_out", default=None)
    main(p.parse_args())
