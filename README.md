# Sereact 3D Bounding Box Prediction (DGCNNBBox)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A modular, production-ready implementation of a **Dynamic Graph CNN (DGCNN)** for 3D bounding box regression directly from point clouds. This project utilizes a novel 7-channel contextual sampling approach to accurately infer bounding boxes even under severe Z-depth occlusion.

---

## 🌟 Key Features

* **Modular Package Architecture:** Cleanly separated package structure (`src/sereact_3d_bbox`) with standalone execution scripts.
* **Context-Aware Sensing (7-Channels):** Samples both the object geometry and surrounding background context (XYZ + RGB + Mask) to counter partial occlusions (e.g., inferring the floor).
* **Robust Preprocessing:** Incorporates **Median Absolute Deviation (MAD)** filtering to gracefully handle depth-sensor bleeding and mask leakage.
* **Continuous 6D Rotation:** Utilizes Gram-Schmidt orthogonalization (Zhou et al. 2019) to ensure valid, continuous SO(3) box rotation matrices without gimbal lock.
* **Production Ready Export:** Includes pipeline to export to **FP32 & INT8 Quantized ONNX** graphs with sub-millisecond CPU/GPU latencies.

---

## 🛠️ Installation

We recommend using a virtual environment (e.g., `venv` or `conda`). The project uses a `pyproject.toml` for automated dependency management.

```bash
# Clone the repository
git clone https://github.com/your-repo/sereact_3d_bbox.git
cd sereact_3d_bbox

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install the project and dependencies in editable mode
pip install -e .
```

---

## 🚀 Usage

All executable entry-points are located inside the `scripts/` directory. 

### 1. Training
Trains the network using Cosine LR scheduling, AMP (Automatic Mixed Precision), and composite Hungarian/Chamfer loss.
```bash
python scripts/train.py \
    --data_root /path/to/dataset \
    --epochs 100 \
    --batch_size 32 \
    --save_path best_model.pth
```

### 2. Evaluation
Evaluates the model on the held-out test split, plots MCD (Mean Corner Distance) histograms, computing `Recall @ Thresholds`, and rendering failure cases.
```bash
python scripts/test.py \
    --data_root /path/to/dataset \
    --checkpoint best_model.pth \
    --vis_dir test_output
```

### 3. Interactive Inference
Runs inference on a specific sample or entire dataset. Outputs interactive 3D **Plotly HTML** files with Ground Truth and Predictions seamlessly overlaid.
```bash
python scripts/inference.py \
    --data_root /path/to/dataset \
    --weights best_model.pth \
    --out_dir output_visualizations
```

### 4. ONNX & INT8 Export
Exports the model to ONNX using Opsets=18 and generates a dynamically quantized INT8 model roughly 4x smaller.
```bash
python scripts/export_onnx.py \
    --checkpoint best_model.pth \
    --out_dir onnx_export
```
*(Once exported to ONNX, you can build TensorRT engines using `trtexec --onnx=onnx_export/dgcnn_bbox.onnx --fp16 --saveEngine=dgcnn_bbox.trt`)*

---

## 🧠 Architecture & Methodology

### The DGCNNBBox Network
Instead of treating points independently (like PointNet), **EdgeConv** layers dynamically calculate a $k$-NN graph at each network stage to merge local and global point arrangements. 

```text
Input: (Batch, 7, 1024)
       ├─ XYZ  (centred on median anchor)
       ├─ RGB  (normalized [0,1])
       └─ Mask (binary: 1=object, 0=bg context)
              │
  ┌───────────▼─────────────────────┐
  │  EdgeConv × 4  (DGCNN Backbone) │
  │  EC1: 7  → 64   (local geom)    │
  │  EC2: 64 → 64   (local feat)    │
  │  EC3: 64 → 128  (mid-range)     │
  │  EC4: 128→ 256  (global)        │
  └───────────┬─────────────────────┘
              ▼
  Conv1d (512→1024) → Global Max-Pool → FC Layers
              │
     ┌────────┼──────────┐
     ▼        ▼          ▼
   center   log_dims   rot6d
   (3,)     (3,)       (6,)
```

### Tri-Partite Loss Function
The model targets 3 separate box representations through a combined loss module (`src/sereact_3d_bbox/models/loss.py`):
1. **Chamfer Loss:** Order-agnostic distance alignment. Highly robust early in training when rotations are stochastic.
2. **Hungarian Smooth-L1:** Strictly computes per-corner losses after bipartite assignment, directly targeting Mean Corner Distance metrics.
3. **Relative Centre Loss:** Predicts origin coordinates relative to the median-filtered anchor, completely eliminating spatial drift vulnerabilities.

### MAD Cleaning Pipeline
Depth sensors inherently suffer from "bleeding" where distant background edges attach to foreground masks. 
We employ a **Median Absolute Deviation (MAD)** filter internally within `dataset.dataset.py`:
- Extracts the median coordinate of the mask.
- Computes deviations from this median.
- Discards geometry points drifting $> 5.0 \times \text{MAD}$.  
This provides a 50% breakdown threshold, meaning the point cloud survives cleanly even if *half* of the mask segments contain leaked background artifacts.

---

## 📂 Project Structure

```text
sereact_3d_bbox/
├── pyproject.toml              # Build system and dependencies
├── README.md                   # Project documentation
├── scripts/
│   ├── train.py                # Main training loop
│   ├── test.py                 # Evaluation & test metrics
│   ├── inference.py            # Local inference & HTML plotting
│   └── export_onnx.py          # Export to FP32 & INT8 ONNX
└── src/
    └── sereact_3d_bbox/
        ├── __init__.py
        ├── data/
        │   ├── __init__.py
        │   └── dataset.py      # Context dataset & MAD Filtering
        ├── models/
        │   ├── __init__.py
        │   ├── dgcnn.py        # EdgeConv & Graph feature components OOM fixed
        │   └── loss.py         # Chamfer, Hungarian, & Anchor loss modules
        └── utils/
            └── __init__.py
```
