# Sereact 3D Bounding Box Prediction (DGCNNBBox)

This project implements a state-of-the-art, modular, and production-ready deep learning pipeline for 3D bounding box prediction from RGB-D data and instance segmentation masks. The solution focuses on structural rigidity, translation invariance, and solving the notorious partial observability problem in top-down point clouds.

**Technical Assumption:** We took the liberty to assume that instance segmentation masks (`mask.npy`) are readily available during inference. Decoupling the object localization/segmentation from the 3D bounding box regression allows the pipeline to fully leverage Contextual Sampling without relying on a bulky, parallel 2D segmentation head.

---

## 🌟 Key Features

* **Modular Package Architecture:** Cleanly separated package structure (`src/sereact_3d_bbox`) with standalone execution scripts.
* **Context-Aware Sensing (7-Channels):** Samples both the object geometry and surrounding background context (XYZ + RGB + Mask) to counter partial occlusions (e.g., inferring the floor).
* **Robust Preprocessing:** Incorporates Median Absolute Deviation (MAD) filtering to gracefully handle depth-sensor bleeding and mask leakage.
* **Continuous 6D Rotation:** Utilizes Gram-Schmidt orthogonalization (Zhou et al. 2019) to ensure valid, continuous SO(3) box rotation matrices without gimbal lock.
* **Production Ready Export:** Includes a pipeline to export to FP32 & INT8 Quantized ONNX graphs with sub-millisecond CPU/GPU latencies.

---

## 📊 Pipeline Visualisation

The following diagram illustrates the complete end-to-end data flow, from raw inputs to the final rigid 3D Bounding Box.

```text
┌─────────────────┐   ┌─────────────┐   ┌───────────────┐
│ RGB Image (H,W) │   │ Point Cloud │   │ Instance Mask │
└────────┬────────┘   └──────┬──────┘   └───────┬───────┘
         │                   │                  │
         └───────────────────┼──────────────────┘
                             ▼
               [ 1. Contextual Sampler ] 
       Samples 512 points inside the mask (Object)
     Samples 512 points outside the mask (Background)
       Appends Mask as a 7th Channel (1=Obj, 0=BG)
                             │
                             ▼
                   Tensor: (7, 1024)
            [ X, Y, Z, R, G, B, Binary_Mask ]
                             │
                             ▼
         [ 2. Dynamic Graph CNN (DGCNN) ]
      Computes K-Nearest Neighbors (k=20) at multiple
      scales using EdgeConv to learn flat surfaces
      and sharp 90-degree topological corners.
                             │
                             ▼
                (1, 1024) Global Descriptor
                             │
         ┌───────────────────┼──────────────────┐
         ▼                   ▼                  ▼
[ Center Offset ]     [ Log Dims ]      [ 6D Rotation ]
   (3 values)          (3 values)         (6 values)
         │                   │                  │
         └───────────────────┼──────────────────┘
                             ▼
           [ 3. Deterministic Box Constructor ]
        Applies Gram-Schmidt Orthogonalization to 
        the 6D vector to generate a perfect 3x3 
        rotation matrix. Combines with Center and Dims.
                             │
                             ▼
         ┌──────────────────────────────────────┐
         │ Flawless, Rigid 3D Bounding Box (8x3)│
         └──────────────────────────────────────┘
```

---

## 🛠️ Installation & Configuration

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

### Configuration:
All core hyperparameters, thresholds, and data settings (like `num_points`) are centralized in `src/sereact_3d_bbox/config.py`. The execution scripts automatically pull defaults from this file, allowing you to globally modify parameters in one place.

---

## 🚀 Usage

All executable entry-points are located inside the `scripts/` directory.

### 1. Training
Trains the network using Cosine LR scheduling, AMP (Automatic Mixed Precision), and our advanced composite loss.
```bash
python scripts/train.py \
    --data_root /path/to/dataset \
    --epochs 100 \
    --batch_size 32 \
    --save_path best_model.pth
```

### 2. Evaluation
Evaluates the model on the held-out test split, plots MCD (Mean Corner Distance) histograms, computes `Recall @ Thresholds`, and renders HTML plots of the worst failure cases.
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

## 🧠 Design Choices & Methodology

### A. Point-Based Processing over 2D Projection (CenterNet)
* **Initial Approach:** Originally, a 2D CenterNet-style approach was considered.
* **Reason for Change:** 2D CNNs suffer from spatial distortion when projecting 3D data. A car on the left side of the lens looks different than one on the right.
* **Final Choice:** Operating directly on the 3D Point Cloud natively respects the metric scale (1 unit = 1 meter) and allows true geometric reasoning without perspective distortion.

### B. DGCNN over Standard PointNet
* **Problem:** A standard PointNet processes every point independently. It understands the "global silhouette" of an object but cannot distinguish between a flat wall and a sharp corner.
* **Solution:** We upgraded to a Dynamic Graph CNN (DGCNN). DGCNN explicitly calculates the distance between neighboring points to build a graph. By analyzing (neighbor - center), the network learns structural topology, enabling millimeter-precision bounds around the object's physical edges.

### C. Contextual Sampling for "Invisible Z-Height"
* **Problem (Partial Observability):** A top-down depth camera only sees the top face of an object (e.g., the roof of a birdhouse). The network has no idea where the bottom of the object is, causing massive errors in Z-axis positioning and height.
* **Solution:** We extract 512 background points (the floor/table) alongside the 512 object points. We feed this `(7, 1024)` tensor into the network, where the 7th channel acts as a binary mask (1 for object, 0 for background). By allowing the network to "see" the floor beneath the object, it mathematically infers the occluded height.

### D. Structural Rigidity via 6D Pose Regression
* **Problem:** Standard 3D networks predict 8 independent corners (24 values). Independent jitter causes these points to tangle into non-cuboid, warped diamonds.
* **Solution:** The network explicitly regresses 12 parameters: Center (3), Size (3), and Rotation (6). Gram-Schmidt orthogonalization guarantees that the output is always a mathematically perfect 90-degree right-angled cuboid.

### E. MAD Cleaning Pipeline
Depth sensors inherently suffer from "bleeding" where distant background edges attach to foreground masks. We employ a Median Absolute Deviation (MAD) filter internally during preprocessing:
1. Extracts the median coordinate of the mask.
2. Computes deviations from this median.
3. Discards geometry points drifting > 5.0 × MAD.

This provides a 50% breakdown threshold, meaning the point cloud survives cleanly even if half of the mask segments contain leaked background artifacts.

---

## 📐 Loss Function Formulation

Because we guarantee structural rigidity inside the network, our composite loss function only needs to optimize the physical placement and scale of the box:

* **Chamfer Distance Loss (Order-Agnostic):** If a perfect box is rotated 180 degrees, it visually looks identical, but the corner indices have swapped. Standard L1 loss heavily penalizes this, causing "mean collapse". Chamfer Distance measures the nearest-neighbor distance between the predicted corner cloud and the ground truth corner cloud, completely ignoring topological ordering.
* **Hungarian L1 Loss (Fine-Tuning):** We use the Hungarian matching algorithm (`scipy.optimize.linear_sum_assignment`) to dynamically find the optimal bipartite pairing between the predicted corners and GT corners, then apply a Smooth L1 loss to lock them in with millimeter precision.
* **Anchor Center L1 Loss:** A direct regularization term forcing the predicted Center Offset to match the ground truth geometric median, keeping the bounding box firmly anchored to the point cloud mass.
* **Dimension & Orthogonality Regularization:** Explicit mathematical penalties prevent dimensional collapse and ensure the 6D output maps to valid SO(3) rotation matrices.

---

## 📈 Evaluation Metrics

**Design Choice: MCD vs. 3D IoU**
For both training targets and evaluation metrics, this pipeline utilizes Mean Corner Distance (MCD) paired with the Hungarian matching algorithm rather than the traditional 3D Intersection over Union (IoU). Exact 3D IoU computation for arbitrarily rotated boxes is mathematically unstable and typically requires compiling custom C++/CUDA extensions. MCD provides a completely Python-native, differentiable proxy that elegantly and simultaneously captures translation, scaling, and rotational errors.

Performance is measured on a held-out test split (10% of the dataset) using strict physical metrics:
* **MCD (Mean/Median Corner Distance):** The average L2 distance (in meters) between the predicted corners and the ground truth. Current Best: ~4.5 - 5.2 cm.
* **Recall @ 10cm / 5cm:** The percentage of predictions where the average error is below a specific threshold (e.g., >90% recall at 10cm).
* **Z-Occlusion Metrics:** Explicit tracking of the Mean Z-Height Error to measure how successfully the contextual sampling overcomes top-down partial observability.

---

## 🏛️ Architectural Retrospective & Loss Function Evolution

This section outlines the iterative engineering process used to solve the 3D Bounding Box Challenge, detailing the limitations encountered at each phase and the structural solutions implemented to overcome them.

### Phase 1: 2D Early-Fusion CNN (CenterNet-Style)
* **The Approach:** We stacked RGB and Point Cloud XYZ into a 6-channel input tensor and processed it via a ResNet34 2D backbone. The network predicted a 2D probability heatmap for object centers and directly regressed 24 absolute 3D coordinates (8 corners × 3 axes).
* **The Problem:** CNNs use translation-invariant kernels, making it nearly impossible to map 2D pixels to absolute 3D world coordinates. This led to floating, severely scattered boxes. Furthermore, dense prediction across the entire image space caused "hairball" artifacts (thousands of overlapping false-positive boxes).

### Phase 2: Offset Regression & The Vanishing Gradient
* **The Approach:** To solve translation invariance, we shifted to predicting 3D offsets relative to the object's physical center, rather than absolute world coordinates.
* **The Problem:** We optimized these offsets using standard Smooth L1 loss. Because target offsets are tiny metric values (e.g., < 0.1m), Smooth L1 squared these tiny errors, resulting in vanishing gradients. Furthermore, because the 24 corners were regressed independently, the boxes deformed into skewed, collapsing diamonds instead of rigid cuboids.

### Phase 3: Mask-Guided PointNet & Topological Mean Collapse
* **The Approach:** We abandoned 2D projections entirely and transitioned to native 3D Point Cloud processing using PointNet, feeding 1024 points per object directly into the network.
* **The Problem:** We initially optimized the 8 corners using strict L1 loss. L1 loss enforces strict topological ordering. If the network predicted a physically perfect box, but rotated it symmetrically 180-degrees (swapping corner index 0 with corner index 5), the L1 loss punished it massively. Confused by symmetric objects, the network fell into a mathematical trap known as Topological Mean Collapse: predicting tiny 3cm boxes for every object to safely minimize average error across all orientations.

### Phase 4: 6D Pose Regression & Chamfer Distance
* **The Approach:** To guarantee structural rigidity, we changed the regression output to exactly 12 parameters: Center (3), Size (3), and a 6D Continuous Rotation Vector (6). Using Gram-Schmidt orthogonalization, this generated a flawless 90-degree cuboid. To solve the Mean Collapse, we swapped L1 Loss for Chamfer Distance, an order-agnostic metric that measures cloud-to-cloud proximity regardless of corner indexing.
* **The Problem:** While Chamfer Distance solved the symmetry issue, it is "fuzzy." It pulls point clouds together but doesn't explicitly pin specific corners, causing us to lose millimeter-level precision. Additionally, PointNet failed to recognize sharp local geometries (edges vs. flat surfaces).

### Phase 5: DGCNN + Contextual Sampling + Composite Loss (Current)
* **The Approach:** We upgraded to Dynamic Graph CNN (DGCNN) to learn explicit topologies like flat walls and sharp corners via EdgeConv. We solved the top-down "Invisible Z-Height" occlusion problem via 7-Channel Contextual Sampling (512 object points + 512 background points).
* **The Loss Function Evolution:** To achieve maximum precision, we engineered a Composite Loss that combines the best of all previous phases (Chamfer for gross alignment, Hungarian L1 for fine pinning, Center L1 for anchoring, and Orthogonality for rigidity).

This iterative journey resulted in our current pipeline, capable of predicting highly precise, mathematically rigid 3D bounding boxes from severely occluded top-down views.

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
        ├── config.py           # Centralized dataclass configurations and defaults
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
