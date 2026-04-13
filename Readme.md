3D Bounding Box Prediction Pipeline

This project implements a state-of-the-art deep learning pipeline for 3D bounding box prediction from RGB-D data and instance segmentation masks. The solution focuses on structural rigidity, translation invariance, and solving the notorious partial observability problem in top-down point clouds.

1. Pipeline Visualisation

The following diagram illustrates the complete end-to-end data flow, from raw inputs to the final rigid 3D Bounding Box.

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


2. Design Choices & Reasoning

A. Point-Based Processing over 2D Projection (CenterNet)

Initial Approach: Originally, a 2D CenterNet-style approach was considered.

Reason for Change: 2D CNNs suffer from spatial distortion when projecting 3D data. A car on the left side of the lens looks different than one on the right.

Final Choice: Operating directly on the 3D Point Cloud natively respects the metric scale (1 unit = 1 meter) and allows true geometric reasoning without perspective distortion.

B. DGCNN over Standard PointNet

Problem: A standard PointNet processes every point independently. It understands the "global silhouette" of an object but cannot distinguish between a flat wall and a sharp corner.

Solution: We upgraded to a Dynamic Graph CNN (DGCNN). DGCNN explicitly calculates the distance between neighboring points to build a graph. By analyzing (neighbor - center), the network learns structural topology, enabling millimeter-precision bounds around the object's physical edges.

C. Contextual Sampling for "Invisible Z-Height"

Problem (Partial Observability): A top-down depth camera only sees the top face of an object (e.g., the roof of a birdhouse). The network has no idea where the bottom of the object is, causing massive errors in Z-axis positioning and height.

Solution: We extract 512 background points (the floor/table) alongside the 512 object points. We feed this (7, 1024) tensor into the network, where the 7th channel acts as a binary mask (1 for object, 0 for background). By allowing the network to "see" the floor beneath the object, it mathematically infers the occluded height.

D. Structural Rigidity via 6D Pose Regression

Problem: Standard 3D networks predict 8 independent corners (24 values). Independent jitter causes these points to tangle into non-cuboid, warped diamonds.

Solution: The network explicitly regresses 12 parameters: Center (3), Size (3), and Rotation (6). Gram-Schmidt orthogonalization guarantees that the output is always a mathematically perfect 90-degree right-angled cuboid.

3. Loss Function Formulation

Because we guarantee structural rigidity inside the network, our loss functions only need to optimize the physical placement of the box.

Chamfer Distance Loss (Order-Agnostic)

Intuition: If a perfect box is rotated 180 degrees, it visually looks identical, but the corner indices (0 and 5) have swapped. Standard L1 loss heavily penalizes this, causing the network to panic and output tiny 3cm boxes ("mean collapse").

Formula: Chamfer Distance measures the nearest-neighbor distance between the predicted corner cloud and the ground truth corner cloud, completely ignoring topological ordering.

Hungarian L1 Loss (Fine-Tuning)

Intuition: We use the Hungarian matching algorithm (scipy.optimize.linear_sum_assignment) to dynamically find the optimal bipartite pairing between the predicted corners and GT corners, then apply a Smooth L1 loss to lock them in.

Anchor Center L1 Loss

Intuition: A direct regularization term forcing the predicted Center Offset to match the ground truth geometric median, keeping the bounding box firmly anchored to the point cloud mass.

4. Evaluation Metrics

Performance is measured on a held-out test split (10% of the dataset) using strict physical metrics:

MCD (Mean/Median Corner Distance): The average L2 distance (in meters) between the predicted corners and the ground truth. Current Best: ~4.5 - 5.2 cm.

Recall @ 10cm / 5cm: The percentage of predictions where the average error is below a specific threshold (e.g., >90% recall at 10cm).

Z-Occlusion Metrics: Explicit tracking of the Mean Z-Height Error to measure how successfully the contextual sampling overcomes top-down partial observability.

5. Getting Started & Execution

The project has been refactored into a standard Python package (`src/sereact_bbox/`) with entry-point scripts stored in `scripts/`. You can run them using the provided `./run.sh` wrapper, or by installing the package in editable mode via `pip install -e .`.

1. Training the DGCNN Model

Trains the model from scratch using the 7-channel Contextual Sampling dataset.

```bash
./run.sh train.py --data_root /path/to/dataset --epochs 80 --batch_size 16 --in_channels 7
```

2. Testing & Evaluation

Evaluates the model on the unseen test set, calculates Hungarian MCD, and isolates the worst-performing predictions (errors > 10cm) for failure analysis.

```bash
./run.sh test.py --data_root /path/to/dataset --checkpoint best_model.pth --vis_dir test_output
```

3. Interactive Visualization (Inference)

Generates fully interactive 3D Plotly HTML files. Open the output files in any web browser to pan, zoom, and rotate around the predicted rigid boxes and raw point clouds.

```bash
./run.sh inference.py --data_root /path/to/dataset --checkpoint best_model.pth --out_dir output
```

4. High-Throughput Deployment (ONNX)

Exports the PyTorch graph to a universally deployable FP32 ONNX model (opset 18), ready for TensorRT or ONNXRuntime ingestion.

```bash
./run.sh export_onnx.py --checkpoint best_model.pth --out_dir onnx_export
```
