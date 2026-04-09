import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import os
import open3d as o3d
# Get the directory where this Python script is located
current_dir = '/home/admin/Downloads'
path = Path(os.path.join(current_dir, 'dl_challenge','8b061a8a-9915-11ee-9103-bbb8eae05561'))

# 1. Load all the data dynamically from the same directory
rgb = mpimg.imread(path / 'rgb.jpg')
masks = np.load(path / 'mask.npy')     # Shape: (3, H, W)
bboxes = np.load(path / 'bbox3d.npy')  # Shape: (3, 8, 3)
pc = np.load(path / 'pc.npy')          # Shape: (3, H, W)
num_objects = masks.shape[0]
cmap = plt.get_cmap('tab10')  # Color map for dynamic coloring

# 2. Prepare the Point Cloud
# The point cloud is currently (3, H, W). We need to flatten it to a list of (X, Y, Z) points
H, W = pc.shape[1], pc.shape[2]
points_3d = pc.reshape(3, -1).T  # Now shape is (H*W, 3)

# Prepare an array for colors, defaulting to a light gray for background points
colors_3d = np.ones((H * W, 3)) * 0.7

# Flatten the masks so they match our flattened point cloud
masks_flat = masks.reshape(num_objects, -1)

# Apply colors to the points that belong to specific objects
for i in range(num_objects):
    color_rgb = cmap(i % 10)[:3]  # Get an RGB color from matplotlib
    object_mask = masks_flat[i]
    colors_3d[object_mask] = color_rgb  # Color the masked points

# Create the Open3D PointCloud object
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_3d)
pcd.colors = o3d.utility.Vector3dVector(colors_3d)

# Create a list to hold all our 3D shapes
geometries = [pcd]

# 3. Prepare the 3D Bounding Boxes
# These indices define which corners connect to form the lines of a box
lines = [
    [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
    [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
    [0, 4], [1, 5], [2, 6], [3, 7]  # Connecting pillars
]

for i, box in enumerate(bboxes):
    color_rgb = cmap(i % 10)[:3]

    # Create an Open3D LineSet for the bounding box
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(box)
    line_set.lines = o3d.utility.Vector2iVector(lines)

    # Apply the corresponding object color to all 12 lines of the box
    line_colors = [color_rgb for _ in range(len(lines))]
    line_set.colors = o3d.utility.Vector3dVector(line_colors)

    geometries.append(line_set)

# 4. Launch the Interactive Viewer
print("Launching interactive viewer...")
print("- Left Click + Drag: Rotate")
print("- Scroll Wheel: Zoom")
print("- Shift + Left Click + Drag: Pan/Move")

o3d.visualization.draw_geometries(geometries, window_name="3D Perception Viewer",
                                  width=1024, height=768)

