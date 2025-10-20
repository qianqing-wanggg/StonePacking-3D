### This is the main file for robotic construction test3
import os
import numpy as np
from place_stone_mortar import place_stones_with_placed_stones
import trimesh
import datetime
import glob
import logging

logger = logging.getLogger(__name__)

import argparse

# read from arguments
parser = argparse.ArgumentParser(description="Process dataset ID.")
parser.add_argument("--dataID", type=str, default="18",
                    help="The ID of the dataset (default: 18)")
args = parser.parse_args()


def sort_stones_by_volume(stones):
    sequence = stones.keys()
    sequence = [x for _,x in sorted(zip([stones[i]['mesh'].volume for i in sequence],sequence),reverse=True)]
    return sequence

def read_new_stones(data_dir,placed_ids = []):
    stones = dict()
    read_sequence = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.obj') == False:
                continue
            stone_id = int(file.split('S')[1].split('.')[0])
            logger.info(f'read new stone id: {stone_id}')
            if stone_id in placed_ids:
                continue
            file_dir = os.path.join(root, file)
            # read mesh file
            mesh = trimesh.load(file_dir)
            # scale the mesh
            scale = 1
            mesh.apply_scale(scale)# !scale the mesh 1 times
            # move the mesh so that the min bounds are non-negative
            T = trimesh.transformations.translation_matrix(mesh.bounds[0]*-1)
            new_mesh = mesh.copy()
            _ = new_mesh.apply_transform(T)
            # get the id of the stone
            read_sequence.append(stone_id)
            stones[stone_id] = dict()
            stones[stone_id]['stone_id'] = stone_id
            stones[stone_id]['mesh'] = new_mesh
    # sort stones by volume
    sequence = sort_stones_by_volume(stones)
    # random sequence
    seed = 1
    np.random.seed(seed)
    np.random.shuffle(sequence)
    return stones,sequence
    

def generate_ground_bound_mesh(ground_mesh_dir,world_size):
    thickness = 1
    # create a cube
    xdim = thickness
    ydim = world_size[1]
    zdim = world_size[2]
    centerx = world_size[0]-xdim/2
    centery = ydim/2
    centerz = zdim/2
    transform = trimesh.transformations.translation_matrix([centerx, centery, centerz])
    cube = trimesh.creation.box(extents=[xdim, ydim, zdim], transform=transform)
    cube.export(ground_mesh_dir+'ground_mesh.obj')
    #create bound mesh at y=0
    xdim = world_size[0]
    ydim = thickness
    zdim = world_size[2]
    centerx = xdim/2
    centery = ydim/2
    centerz = zdim/2
    transform = trimesh.transformations.translation_matrix([centerx, centery, centerz])
    cube = trimesh.creation.box(extents=[xdim, ydim, zdim], transform=transform)
    cube.export(ground_mesh_dir+'bound_mesh_y0.obj')
    #create bound mesh at y=world_size[1]
    world_size_1_modified  = world_size[1]
    xdim = world_size[0]
    ydim = thickness
    zdim = world_size[2]
    centerx = xdim/2
    centery = world_size_1_modified-ydim/2
    centerz = zdim/2
    transform = trimesh.transformations.translation_matrix([centerx, centery, centerz])
    cube = trimesh.creation.box(extents=[xdim, ydim, zdim], transform=transform)
    cube.export(ground_mesh_dir+'bound_mesh_y1.obj')
    #create bound mesh at z=0
    xdim = world_size[0]
    ydim = world_size[1]
    zdim = thickness
    centerx = xdim/2
    centery = ydim/2
    centerz = zdim/2
    transform = trimesh.transformations.translation_matrix([centerx, centery, centerz])
    cube = trimesh.creation.box(extents=[xdim, ydim, zdim], transform=transform)
    cube.export(ground_mesh_dir+'bound_mesh_z0.obj')
    #create bound mesh at z=world_size[2]
    xdim = world_size[0]
    ydim = world_size[1]
    zdim = thickness
    centerx = xdim/2
    centery = ydim/2
    centerz = world_size[2]-zdim/2
    transform = trimesh.transformations.translation_matrix([centerx, centery, centerz])
    cube = trimesh.creation.box(extents=[xdim, ydim, zdim], transform=transform)
    cube.export(ground_mesh_dir+'bound_mesh_z1.obj')


def generate_from_placed_stones(layer_i,max_height,placed_stones_dir,available_stones_dir,world_size_meter,ground_mesh_dir,pitch = 1):
    #load ground and bound
    #check if ground_mesh_dir is empty
    if not os.path.exists(ground_mesh_dir):
        os.mkdir(ground_mesh_dir)
        generate_ground_bound_mesh(ground_mesh_dir,world_size_meter)
    ground_mesh = trimesh.load(ground_mesh_dir+'ground_mesh.obj')
    bound_meshes = []
    bound_mesh_files = glob.glob(ground_mesh_dir+'*bound*')
    for bound_mesh_file in bound_mesh_files:
        bound_meshes.append(trimesh.load(bound_mesh_file))

    # load placed stones
    placed_stones = dict()
    placed_ids=[]
    #load available stones
    stones,sequence = read_new_stones(available_stones_dir,placed_ids)

    for stone_object in stones.values():
        # a dictonary indexed by stone id
        stones[stone_object['stone_id']] = stone_object
    # voxelize
    world_size = (int(world_size_meter[0]/pitch),int(world_size_meter[1]/pitch),int(world_size_meter[2]/pitch))
    #get the time date stamp
    time_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    result_dir = f"../result/example"+f"{args.dataID}/wall_size_{world_size[0]}_{world_size[1]}_{world_size[2]}_timestamp_{time_stamp}"
    if not os.path.exists('../result/'):
        os.mkdir('../result/')
    if not os.path.exists("../result/example"+f"{args.dataID}/"):
        os.mkdir("../result/example"+f"{args.dataID}")
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    logging.basicConfig(filename=result_dir+'/log.txt', level=logging.DEBUG)
    rotation_angle_options = [0]#angles in degree
    nb_processor = 12
    nb_cand = 1#2
    
    place_stones_with_placed_stones(layer_i,nb_cand = nb_cand,result_dir=result_dir, sequence=sequence, stones=stones, \
        placed_stones = placed_stones,wall_size=world_size,ground_mesh = ground_mesh,bound_meshes = bound_meshes,\
            rotation_angle_options = rotation_angle_options,nb_processor = nb_processor,pitch = pitch,max_height = max_height)




if __name__ == '__main__':
    layer_i = 1
    max_height = 1
    data_dir = "../data/example"+f"{args.dataID}/"
    placed_stones_dir = "../data/example"+f"{args.dataID}/placed_stones_asbuilt/"
    available_stones_dir = "../data/example"+f"{args.dataID}/available_stones_rot/"
    #world_size_meter = (100,200,125)
    world_size_meter = (70+1,70+2,40+2)
    ground_mesh_dir = "../data/example"+f"{args.dataID}/ground_bound/"
    pitch = 1

    
    generate_from_placed_stones(layer_i,max_height,placed_stones_dir,available_stones_dir,world_size_meter,ground_mesh_dir,pitch = pitch)

