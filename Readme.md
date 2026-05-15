# 3D Bounding Box Prediction Pipeline

This project implements a state-of-the-art deep learning pipeline for 3D bounding box prediction from RGB-D data and instance segmentation masks. The solution focuses on structural rigidity, translation, and rotation invariance to guarantee accurate and mathematically valid 3D bounding boxes.

---

## 1. Pipeline Visualization

The following diagram illustrates the complete end-to-end data flow, from raw inputs to the final rigid 3D bounding box:

```
┌─────────────────┐   ┌─────────────┐   ┌───────────────┐
│ RGB Image (H,W) │   │ Point Cloud │   │ Instance Mask │
└────────┬────────┘   └──────┬──────┘   └───────┬───────┘
         │                   │                  │
         └───────────────────┼──────────────────┘
                             ▼
               [ 1. Contextual Sampler ] 
     Samples 512 points inside the mask (Object)
     Samples 512 points outside the mask (Background)
     Appends Mask as a 7th Channel (1 = Obj, 0 = BG)
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
        rotation matrix. Combines with center and dims.
                             │
                             ▼
         ┌──────────────────────────────────────┐
         │  Flawless, Rigid 3D Bounding Box    │
         │        (Shape: 8×3)                 │
         └──────────────────────────────────────┘
```

---

## 2. Design Choices & Reasoning

### A. Point-Based Processing over 2D Projection (CenterNet)

- **Initial Approach:** A 2D CenterNet-style method using RGB images.
- **Reason for Change:** 2D CNNs distort spatial information when projecting 3D data; objects appear differently based on lens position, introducing geometric inconsistencies.
- **Final Choice:** Operate directly on the 3D point cloud. This preserves real-world metric scale (1 unit = 1 meter) and enables true geometric reasoning.

### B. DGCNN over Standard PointNet

- **Problem:** PointNet processes every point independently—good for shape, bad for details. It cannot distinguish flat walls from sharp corners.
- **Solution:** DGCNN builds a dynamic graph of point neighborhoods, capturing local geometric features (like corners) by analyzing point relationships.

### C. Contextual Sampling for "Invisible Z-Height"

- **Problem (Partial Observability):** A top-down camera sees only the top face (e.g., birdhouse roof)—the base is hidden, which leads to Z-height ambiguity.
- **Solution:** Extract both object (512) and background (512) points. A 7th binary mask channel distinguishes object/background, allowing the network to learn object-ground separation and Z-inference.

### D. Structural Rigidity via 6D Pose Regression

- **Problem:** Traditional networks predict 8 independent corners (24 values), resulting in non-cuboidal and warped outputs due to uncontrolled jitter.
- **Solution:** Regress only 12 parameters: center (3), size (3), rotation (6). Gram-Schmidt orthogonalization guarantees a mathematically perfect 90-degree cuboid.

---

## 3. Loss Function Formulation

Structural rigidity guarantees simplify losses; these focus only on placement:

**Chamfer Distance Loss (Order-Agnostic):**
- Intuition: Two perfect cuboids, even if rotated 180°, are identical in shape. Chamfer distance ignores corner order, measuring closest-point distances only.

**Hungarian L1 Loss (Fine-Tuning):**
- Uses the Hungarian matching algorithm to optimally pair predicted and ground-truth corners, then applies robust L1 loss on pairs.

**Anchor Center L1 Loss:**
- Directly regularizes the predicted center to match the ground-truth geometric median, ensuring tight box-point cloud anchoring.

---

## 4. Evaluation Metrics

- **MCD (Mean/Median Corner Distance):** Average L2 distance (meters) between predicted and ground-truth corners. Best: ~4.5–5.2 cm.
- **Recall @ 10cm / 5cm:** Percentage of objects where error is below threshold (e.g., >90% recall at 10 cm).
- **Z-Occlusion Metrics:** Explicitly tracks mean Z-height error to measure how well contextual sampling overcomes visibility limitations.

---

## 5. Getting Started & Execution

### 1. Training the DGCNN Model

Train from scratch using the 7-channel contextual sampling dataset:
```bash
python train.py --data_root /path/to/dataset --epochs 80 --batch_size 16 --in_channels 7
```

### 2. Testing & Evaluation

Evaluate on the test set (calculates Hungarian MCD, isolates failures):
```bash
python test.py --data_root /path/to/dataset --checkpoint best_model.pth --vis_dir test_output
```

### 3. Interactive Visualization (Inference)

Generate fully interactive 3D Plotly HTML visualizations:
```bash
python inference.py --data_root /path/to/dataset --checkpoint best_model.pth --out_dir output
```
Open output HTML files in your browser for pan, zoom, and 3D inspection.

### 4. High-Throughput Deployment (ONNX)

Export the PyTorch model to ONNX for deployment:
```bash
python export_onnx.py --checkpoint best_model.pth --out_dir onnx_export
```
Ready for TensorRT or ONNXRuntime inference.

---

Feel free to reach out with questions or to contribute!
