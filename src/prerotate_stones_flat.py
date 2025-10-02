import os, time
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.spatial.distance import cdist,pdist


start_time = time.time()
# Input and output folder
available_stones_dir = "../data/example18/available_stones/"
output_dir= "../data/example18/available_stones_rot/"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# Iterate all stones in the input folder with id from 0 to 150
for stone_i in range(147,170):
    # Find S000.obj
    stone_number = f"{stone_i:0{3}d}"
    print("Stone ",stone_number)
    STONE = available_stones_dir+"stone_"+stone_number+".obj"
    #new_stone_number = f"{stone_i+147:0{3}d}"
    new_stone_number = stone_number

    if not os.path.exists(STONE):
        continue

    # Load a triangle mesh
    mesh = o3d.io.read_triangle_mesh(STONE)

    # Compute vertex normals
    mesh.compute_vertex_normals()

    # Get the optimal bbox
    obb = mesh.get_oriented_bounding_box()
    extent = obb.extent  # [length, width, height] along the box's local axes

    # Find index of smallest extent and its direction in world space
    min_axis_index = np.argmin(extent)
    obb_axes = obb.R  # 3x3 matrix: columns are the OBB's local axes
    source_axis = obb_axes[:, min_axis_index]

    # Target axis = global x-axis
    target_axis = np.array([1.0, 0.0, 0.0])

    # Compute rotation matrix to align source_axis to target_axis
    v = np.cross(source_axis, target_axis)
    s = np.linalg.norm(v)
    c = np.dot(source_axis, target_axis)

    # Handle near-alignment or exact alignment cases
    if s < 1e-8 and c > 0.9999:
        R_align = np.eye(3)
    else:
        vx = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
        R_align = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2 + 1e-10))

    # Rotate mesh around its center
    mesh.rotate(R_align, center=mesh.get_center())

    # Assuming `new_mesh` is an open3d.geometry.TriangleMesh object
    bbox = mesh.get_axis_aligned_bounding_box()
    bbox_min = np.asarray(bbox.min_bound)  # Get the min bounds of the AABB

    # Compute translation matrix to move the mesh so its min_bound is at (0, 0, 0)
    to_positive = np.eye(4)  # Identity matrix
    to_positive[:3, 3] = -bbox_min  # Set translation part of the matrix

    # Apply transformation
    mesh.transform(to_positive)

    # write mesh to file
    o3d.io.write_triangle_mesh(output_dir+"S"+new_stone_number+".obj",mesh)
