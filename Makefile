PYTHON ?= python3
CHECKPOINT ?= best_model.pth

.PHONY: help install install-dev lint test build check preflight train evaluate gallery inference export

help:
	@printf "Available targets:\n"
	@printf "  install      Install runtime and optional workflow dependencies\n"
	@printf "  install-dev  Install development dependencies\n"
	@printf "  check        Run lint, tests, compilation, and package build\n"
	@printf "  preflight    Validate the configured dataset before training\n"
	@printf "  train        Train the default V2 model\n"
	@printf "  evaluate     Evaluate CHECKPOINT on the held-out split\n"
	@printf "  gallery      Refresh README evaluation images\n"
	@printf "  inference    Run interactive inference using CHECKPOINT\n"
	@printf "  export       Export CHECKPOINT to ONNX\n"

install:
	$(PYTHON) -m pip install -e ".[all]"

install-dev:
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	$(PYTHON) -m ruff check .

test:
	$(PYTHON) -m pytest

build:
	$(PYTHON) -m build

check: lint test
	$(PYTHON) -m compileall -q scripts src tests
	$(PYTHON) -m build

preflight:
	$(PYTHON) scripts/validate_dataset.py

train:
	$(PYTHON) scripts/train.py --model_version v2

evaluate:
	$(PYTHON) scripts/test.py --checkpoint "$(CHECKPOINT)"

gallery:
	$(PYTHON) scripts/test.py --checkpoint "$(CHECKPOINT)" --readme_gallery_dir docs/images/evaluation

inference:
	$(PYTHON) scripts/inference.py --weights "$(CHECKPOINT)"

export:
	$(PYTHON) scripts/export_onnx.py --checkpoint "$(CHECKPOINT)"
