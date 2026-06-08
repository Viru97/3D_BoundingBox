# Contributing

## Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
make install-dev
cp paths.example.json paths.local.json
```

Update `paths.local.json` for your machine. Keep datasets, checkpoints, exports, and generated visualizations out of git.

## Before Opening a Pull Request

```bash
make check
```

Changes to preprocessing, splitting, loss functions, or decoding should include a focused regression test. Accuracy changes should also report held-out metrics from the committed group-disjoint split manifest.

## Evaluation Gallery

Refresh the tracked README images only when intentionally updating the documented checkpoint results:

```bash
make gallery CHECKPOINT=best_model_pose.pth
```
