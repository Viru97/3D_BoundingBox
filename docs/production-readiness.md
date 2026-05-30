# Production Readiness

This repository is engineered as a reproducible research baseline and integration-ready package. The current model checkpoint is not approved for production decisions.

## Supported Workflow

1. Validate dataset structure with `make preflight`.
2. Train with a committed split manifest and versioned code revision.
3. Evaluate the selected checkpoint on the held-out split.
4. Export ONNX and require PyTorch/ONNX parity to pass.
5. Store the checkpoint, split manifest, evaluation JSON, and code revision together in an artifact registry.

## Current Accuracy Gate

The documented checkpoint does not meet the project stretch targets:

| Metric | Current | Stretch target |
| --- | ---: | ---: |
| Median MCD | 7.82 cm | <= 3.00 cm |
| Recall @ 5 cm | 32.5% | Improve by at least 10 points over baseline |
| Mean angular error | 66.07 deg | <= 15.00 deg |

Do not describe this checkpoint as industry-ready until domain-specific acceptance criteria are agreed and met on representative, group-disjoint data.

## Deployment Checklist

- Add an owner-approved `LICENSE` file before third-party distribution.
- Use a controlled source for `.pth` and `.onnx` artifacts.
- Pin the exact checkpoint and code revision.
- Validate camera calibration and input units in the deployment environment.
- Monitor rejected masks and preprocessing errors.
- Record latency on the target hardware.
- Run ONNX parity checks after every export.
- Add detector or segmenter validation before deploying without trusted `mask.npy` files.
