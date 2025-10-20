import os
import argparse
import open3d as o3d

# read from arguments
parser = argparse.ArgumentParser(description="Process dataset ID.")
parser.add_argument("--dataID", type=str, default="18",
                    help="The ID of the dataset (default: 18)")
args = parser.parse_args()

input_stones_dir = f"../data/example{args.dataID}/stones/"
output_stones_dir = f"../data/example{args.dataID}/available_stones/"
if not os.path.exists(output_stones_dir):
    os.mkdir(output_stones_dir)


def rescale(start=0, end=170, scale = 100):
    for i in range(start, end + 1):
        stone_name = input_stones_dir+f"stone_{i:03d}.obj"
        if os.path.exists(stone_name):
            # Load the mesh
            mesh = o3d.io.read_triangle_mesh(stone_name)
            # Rescale the mesh
            scale_factor = 100
            # This scales the mesh *relative to a center point*, usually the origin
            mesh.scale(scale_factor, center=mesh.get_center())

            # Optional: Save or visualize the rescaled mesh
            o3d.io.write_triangle_mesh(output_stones_dir+f"stone_{i:03d}.obj", mesh)
        else:
            print(f"File not found: {stone_name}")

# Run the scaling
rescale()