import os
import cv2
import numpy as np
import trimesh
from plyfile import PlyData, PlyElement
import time
from multiprocessing import Pool
import scipy.ndimage
import logging
base_pixel_value = 255
left_bound_pixel_value = 254
right_bound_pixel_value = 253
logger = logging.getLogger(__name__)

global global_result_dir

def rotate_face_up(stone_mesh, degree,move_to_positive = True,return_matrix = False):
    new_mesh = stone_mesh.copy()
    centroid = new_mesh.centroid
    to_origin = trimesh.transformations.translation_matrix([-centroid[0],-centroid[1],-centroid[2]])
    _ = new_mesh.apply_transform(to_origin)
    
    if degree>0:
        new_mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2*degree/90, [1,0,0]))
    elif degree<0:
        # first flip the mesh upside down
        new_mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [0,1,0]))
        new_mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2*abs(degree)/90, [1,0,0]))
    if move_to_positive:
        bbox_min = new_mesh.bounds[0]
        to_positive = trimesh.transformations.translation_matrix([-bbox_min[0],-bbox_min[1],-bbox_min[2]])
        _ = new_mesh.apply_transform(to_positive)
    return new_mesh

def get_phi_distance_3d(base):
    # # compute the minimal distance from each pixel of the stone to non-zero pixels of base
    base_reverted  = np.where(base!=0,0,1)
    new_phi = scipy.ndimage.distance_transform_edt(base_reverted, return_distances=True,
                                         return_indices=False)
    return new_phi

def get_proximity_metric_3d(base_phi, brick_bounding_box_matrix_mirrow):
    shift_to = [(brick_bounding_box_matrix_mirrow.shape[0]-1) // 2,(brick_bounding_box_matrix_mirrow.shape[1]-1) // 2,(brick_bounding_box_matrix_mirrow.shape[2]-1) // 2]
    proximity_metric = scipy.ndimage.convolve(
       base_phi, brick_bounding_box_matrix_mirrow, mode='constant', cval=1.0, origin=shift_to)
    #proximity_metric = np.multiply(base_phi, stone)
    return proximity_metric

def get_height_3d(matrix):
    # matrix where the last row in dimension 0 is 0, the others are 1
    height_matrix = np.ones_like(matrix)
    height_matrix[-1,:,:] = 0
    height_matrix = scipy.ndimage.distance_transform_edt(height_matrix, return_distances=True, return_indices=False)
    # normalize
    height_matrix = height_matrix/np.max(height_matrix)
    # cubic
    #height_matrix = height_matrix**3
    return height_matrix



def voxelize_mesh(stone_mesh,pitch = 1,add_minimum_ball = True):

    # upsample vertices of the stone mesh using subdivision
    points,_ = trimesh.remesh.subdivide_to_size(stone_mesh.vertices, stone_mesh.faces,max_edge = pitch/2)

    #points = np.array(stone_mesh.vertices)
    #round_points = occ2points(points)
    round_points = np.round(points/pitch).astype(int)
    #remove duplicate points
    round_points = np.unique(round_points,axis=0)
    
    bbox_min = round_points.min(axis=0)
    if any(bbox_min)<0:
        return False
    bbox_max = round_points.max(axis=0)#included
    #print("round_points max: ",bbox_max)
    #print("round_points min: ",bbox_min)
    stone_l_1 = bbox_max[0]-bbox_min[0]+1
    stone_l_2 = bbox_max[1]-bbox_min[1]+1
    stone_l_3 = bbox_max[2]-bbox_min[2]+1
    #print("stone_l_1: ", stone_l_1)

    #crop the stone matrix
    stone_matrix = np.zeros((stone_l_1,stone_l_2,stone_l_3))
    round_points_to_origin = round_points-bbox_min
    stone_matrix[round_points_to_origin[:,0],round_points_to_origin[:,1],round_points_to_origin[:,2]] = 1

    # add a minimul ball in the stone matrix
    if add_minimum_ball:
        center_stone = round_points_to_origin.mean(axis=0)
        distance_from_center = np.linalg.norm(round_points_to_origin-center_stone,axis=1)
        minimum_ball_radius = np.min(distance_from_center)
        # point inside the ball are set to 1
        all_points = np.argwhere(stone_matrix==0)
        distance_to_center = np.linalg.norm(all_points-center_stone,axis=1)
        stone_matrix[all_points[distance_to_center<minimum_ball_radius][:,0],all_points[distance_to_center<minimum_ball_radius][:,1],all_points[distance_to_center<minimum_ball_radius][:,2]] = 1
    return stone_matrix,[stone_l_1,stone_l_2,stone_l_3],bbox_min

def convert_mesh_to_image(stone_mesh,image_size,pitch = 1, add_minimum_ball = True):
    #check if stone_mesh is an empty mesh
    #print(stone_mesh.vertices.shape)
    if stone_mesh.vertices.shape[1] == 0:
        return np.zeros(image_size)
    stone_matrix,[stone_l_1,stone_l_2,stone_l_3],bbox_min = voxelize_mesh(stone_mesh,pitch = pitch,add_minimum_ball=add_minimum_ball)
    #bbox_min = stone_mesh.bounds[0]
    stone_dimensions = [stone_l_1,stone_l_2,stone_l_3]
    # print("stone_l_1",stone_l_1)
    # print("stone_l_2",stone_l_2)
    # print("stone_l_3",stone_l_3)
    # print("bbox_min",bbox_min)
    # encode the image of the stone
    stone_image = np.zeros(image_size)
    image_start_index = bbox_min
    stone_start_index = [0,0,0]
    image_end_index = [bbox_min[i]+stone_dimensions[i] for i in range(3)]
    stone_end_index = [stone_l_1,stone_l_2,stone_l_3]
    for i in range(3):
        if bbox_min[i]<0:
            exceeding_length = abs(bbox_min[i])
            stone_dimensions[i]-=exceeding_length
            image_start_index[i] = 0
            stone_start_index[i] = exceeding_length
            image_end_index[i] = image_start_index[i]+stone_dimensions[i]
            stone_end_index[i] = stone_start_index[i]+stone_dimensions[i]
        if image_end_index[i]>stone_image.shape[i]:
            exceeding_length = image_end_index[i]-stone_image.shape[i]
            image_end_index[i] = stone_image.shape[i]
            stone_dimensions[i]-=exceeding_length
            stone_end_index[i] = stone_start_index[i]+stone_dimensions[i]
    
    # limit_1 = min(stone_image.shape[0],bbox_min[0]+stone_l_1)
    # limit_2 = min(stone_image.shape[1],bbox_min[1]+stone_l_2)
    # limit_3 = min(stone_image.shape[2],bbox_min[2]+stone_l_3)
    # stone_image[bbox_min[0]:limit_1,\
    #             bbox_min[1]:limit_2,\
    #                 bbox_min[2]:limit_3] = \
    #     stone_matrix[0:limit_1-bbox_min[0],0:limit_2-bbox_min[1],0:limit_3-bbox_min[2]]
    stone_image[image_start_index[0]:image_end_index[0],image_start_index[1]:image_end_index[1],\
                image_start_index[2]:image_end_index[2]] = stone_matrix[stone_start_index[0]:stone_end_index[0],\
                stone_start_index[1]:stone_end_index[1],stone_start_index[2]:stone_end_index[2]]
    
    return stone_image




def add_stone_3d(wall, wall_other_matrix,stone_mesh,p=None,relaxed_mason_criteria = False,pitch = 1):
    global left_bound_pixel_value,right_bound_pixel_value,global_result_dir

    wall_seg_matrix = wall_other_matrix['wall_seg_matrix']
    edge_face_matrix_2 = wall_other_matrix['edge_face_matrix_2']
    edge_face_matrix_3 = wall_other_matrix['edge_face_matrix_3']
    interlocking_distance_2 = wall_other_matrix['interlocking_distance_2']
    interlocking_distance_3 = wall_other_matrix['interlocking_distance_3']

    #voxelize the mesh
    stone_matrix = voxelize_mesh(stone_mesh,pitch = pitch)
    stone_matrix = stone_matrix[0]
    # compute kernel for convolution
    brick_bounding_box_matrix = stone_matrix
    brick_bounding_box_matrix_mirrow = np.flip(np.flip(
        np.flip(brick_bounding_box_matrix, axis=0), axis=1), axis=2)
    shift_to = [(brick_bounding_box_matrix_mirrow.shape[0]-1) // 2,(brick_bounding_box_matrix_mirrow.shape[1]-1) // 2,(brick_bounding_box_matrix_mirrow.shape[2]-1) // 2]
    # non overlapping mask
    non_overlap_mask = np.where(scipy.ndimage.convolve(
        wall, brick_bounding_box_matrix_mirrow, mode='constant', cval=1.0, origin=shift_to) == 0, 1, 0).astype(np.uint8)
    
    # floating positions
    wall_with_ground_and_placed_stones = \
        np.where((wall_seg_matrix!=left_bound_pixel_value)&(wall_seg_matrix!=right_bound_pixel_value)&(wall_seg_matrix!=0),\
                 1,0).astype(np.uint8)
    kernel_dilation_y = np.ones((2, 1,1), np.uint8)
    contour_up = cv2.dilate(wall_with_ground_and_placed_stones, kernel_dilation_y,anchor=(0,0),iterations=1)
    # save voxel
    #save_voxel(contour_up,pitch,global_result_dir,f'contour_up.ply')
    overlap_contour_mask = np.where(scipy.ndimage.convolve(contour_up, brick_bounding_box_matrix_mirrow, mode='constant', cval=0.0,origin = shift_to)>=1,1,0)
    
    # feasible regions
    region_potential = np.multiply(non_overlap_mask, overlap_contour_mask)
        
    #check if there is any feasible region
    if len(np.argwhere(region_potential!=0))==0:
        print("No feasible region with constraint on overlapping and contact")
        return [0,0,0], {"interlocking":-np.inf,"neighbor_height":np.inf,"optimization_score":-np.inf,"distance_to_bound":np.inf}



    #inside bounding box ---  hard requirement
    # wall
    INSIDE = True#!ablation study
    #INSIDE= False
    
    wall_seg_matrix_no_bound = np.where((wall_seg_matrix!=left_bound_pixel_value)&(wall_seg_matrix!=right_bound_pixel_value)&(wall_seg_matrix!=base_pixel_value),wall_seg_matrix,0)
    flag_inside_bounding_box = False
    if np.argwhere(wall_seg_matrix_no_bound!=0).shape[0]!=0:
        max_0_wall_without_bounding = np.min(np.argwhere(wall_seg_matrix_no_bound!=0),axis = 0)[0]
        bounding_box = np.zeros_like(wall_seg_matrix_no_bound)
        bounding_box[0:max_0_wall_without_bounding,:,:] = 1
        #70% of the stone
        portion_30_height = int(stone_matrix.shape[0]*0) 
        brick_bounding_box_matrix = stone_matrix
        #brick_bounding_box_matrix[0:portion_30_height,:,:] = 0
        brick_bounding_box_matrix_mirrow = np.flip(np.flip(
            np.flip(brick_bounding_box_matrix, axis=0), axis=1), axis=2)
        # non overlapping mask
        inside_bounding_box = np.where(scipy.ndimage.convolve(
            bounding_box, brick_bounding_box_matrix_mirrow, mode='constant', cval=1.0, origin=shift_to) == 0, 1, 0).astype(np.uint8)
        outside_bounding_box_volume = scipy.ndimage.convolve(
            bounding_box, brick_bounding_box_matrix_mirrow, mode='constant', cval=1.0, origin=shift_to)
        if INSIDE and len(np.argwhere(np.multiply(region_potential,inside_bounding_box)!=0))!=0:
            region_potential = np.multiply(region_potential,inside_bounding_box)
            flag_inside_bounding_box = True
            print("INSIDE filter applied")
    else:
        outside_bounding_box_volume = np.ones_like(wall_seg_matrix_no_bound)*np.sum(stone_matrix)
        inside_bounding_box = np.zeros_like(wall_seg_matrix_no_bound)
   

    
    # # find how many stones are in touch with the current
    stone_width_dim2 = brick_bounding_box_matrix.shape[1]
    stone_width_dim3 = brick_bounding_box_matrix.shape[2]
    stone_height = brick_bounding_box_matrix.shape[0]

    
    
    flag_good_interlocking_dim2 = True
   
    mason_criteria = np.ones_like(region_potential)
    
    if relaxed_mason_criteria:
        mason_criteria = np.ones_like(region_potential)
   
    if len(np.argwhere(np.multiply(region_potential,mason_criteria)!=0))==0:
        print("No feasible region with constraint on mason's criteria")
        return [0,0,0], {"interlocking":-np.inf,"neighbor_height":np.inf,"optimization_score":-np.inf,"distance_to_bound":np.inf}
   
    # proximity map
    base_phi = get_phi_distance_3d(wall_seg_matrix)
    
    # height map
    base_height = get_height_3d(wall)
    if p is not None:
        #weight_height = p*stone_width*stone_height*10
        #weight_bound = p*wall.shape[0]*0
        weight_height = p*np.sqrt(stone_width_dim2**2+stone_width_dim3**2+stone_height**2)*0
    else:
        weight_height = np.sqrt(stone_width_dim2**2+stone_width_dim3**2+stone_height**2)*0
        #weight_bound = wall.shape[0]*0
    #score_optimization_map = -base_phi
    score_optimization_1 = get_proximity_metric_3d(-base_phi, brick_bounding_box_matrix_mirrow)
    #version 2
    # # distance to origin
    volumne_stone = np.sum(stone_matrix)
    base_matrix_origin = np.zeros_like(wall_seg_matrix)
    base_matrix_origin[0,0,0] = 1
    stone_center_matrix = np.zeros_like(brick_bounding_box_matrix_mirrow)
    stone_center = np.average(np.argwhere(stone_matrix!=0),axis=0)
    stone_center_matrix[int(stone_center[0]),int(stone_center[1]),int(stone_center[2])] = 1
    distance_to_origin = get_phi_distance_3d(base_matrix_origin)
    wall_dim_length = (wall.shape[0]+wall.shape[1]+wall.shape[2])/3
    stone_dim_length = (stone_matrix.shape[0]+stone_matrix.shape[1]+stone_matrix.shape[2])/3
    weight_DiO = stone_dim_length*0.001*volumne_stone/wall_dim_length
    score_optimization_2 = get_proximity_metric_3d(-weight_DiO*distance_to_origin, stone_center_matrix)
    

    # inside bounding box ratio
    weight_bbox = stone_dim_length*0.01*volumne_stone
    score_optimization_3 = -weight_bbox*outside_bounding_box_volume/volumne_stone

    score_optimization = score_optimization_1+score_optimization_2+score_optimization_3
    #score_optimization = score_optimization_1
    JOINTS = False
    if JOINTS:
        score_potential = np.where((region_potential != 0) & (mason_criteria!=0), score_optimization, -np.inf)
    else:
        score_potential = np.where(region_potential != 0, score_optimization, -np.inf)

    best_score = np.max(score_potential)
    best_loc = np.argwhere(score_potential == best_score)[0]
   
    score_optimization_gradient = np.gradient(score_optimization_1+score_optimization_2+score_optimization_3)
   
    # get the direction of refinement for the best location
    best_refinement_direction = [np.sign(score_optimization_gradient[0][best_loc[0],best_loc[1],best_loc[2]]),
                                 np.sign(score_optimization_gradient[1][best_loc[0],best_loc[1],best_loc[2]]),
                                    np.sign(score_optimization_gradient[2][best_loc[0],best_loc[1],best_loc[2]])]



    return best_loc, {"interlocking":-np.inf,"neighbor_height":-np.inf,"optimization_score":score_optimization[best_loc[0],best_loc[1],best_loc[2]],"distance_to_bound":-np.inf,"refinement_direction":best_refinement_direction}

def transform(stone, location):
    shape = stone.shape
    transformed_stone = np.zeros_like(stone)
    transformed_stone[int(location[0]):, int(
        location[1]):] = stone[:shape[0]-int(location[0]), : shape[1]-int(location[1])]
    return transformed_stone


def get_best_placement(wall, wall_seg_matrixs,stone, rotation_angle_options,\
                       elems = {}, contps = {}, weight_height = 1,\
                        func = add_stone_3d, rotation_function = rotate_face_up,\
                            nb_processor = 1,relaxed_mason_criteria = False,iteration = 0,\
                                result_dir = None,pitch = 1,wall_mesh = None,kinematic_check = True):
    wall_seg_matrix = wall_seg_matrixs['wall_seg_matrix']
    best_rotate_pose_index = 0
    best_score = -np.inf
    best_loc = None
    best_evaluation = {"interlocking":0,"neighbor_height":np.inf}
    best_direction = [None,None,None]
    if nb_processor == 1:
        for rotate_pose_index,rotate_sequence in enumerate(rotation_angle_options):
            #rotate stone mesh along x axis
            stone_mesh = rotation_function(stone,rotate_sequence)
            best_loc_this_pose, evaluation_this_pose= func(
                wall, wall_seg_matrixs,stone_mesh,p=weight_height,relaxed_mason_criteria=relaxed_mason_criteria,pitch = pitch)
            

            best_score_this_pose = evaluation_this_pose["optimization_score"]
            print("Score with rotation {} is {}, at {}".format(rotate_sequence,best_score_this_pose,best_loc_this_pose))
            if best_score_this_pose == -np.inf:
                continue
            best_direction_this_pose = evaluation_this_pose["refinement_direction"]
            stone_ = stone_mesh
            
        
            # add stone to wall
            stone_to_add = stone_mesh
            best_loc_this_pose_pitch = [best_loc_this_pose[i]*pitch for i in range(3)]
            T = trimesh.transformations.translation_matrix(best_loc_this_pose_pitch)
            _ = stone_to_add.apply_transform(T)
            stone_image = convert_mesh_to_image(stone_to_add,wall.shape,pitch = pitch)

            
            if  best_score_this_pose> best_score:
                best_rotate_pose_index = rotate_pose_index
                best_score = best_score_this_pose
                best_loc = best_loc_this_pose
                best_direction = best_direction_this_pose
    else:
        # parallel search
        
        for rotation_index_this_processor in range(0,len(rotation_angle_options),nb_processor):
            inputs = []
            for j in range(rotation_index_this_processor,min(rotation_index_this_processor+nb_processor,len(rotation_angle_options))):
                rotate_sequence = rotation_angle_options[j]
                stone_mesh = rotation_function(stone,rotate_sequence)
                #stone_mesh.export(result_dir+f'/iteration_{iteration}_rotation_{j}.ply')
                input_parallel =(wall,wall_seg_matrixs,stone_mesh,weight_height,relaxed_mason_criteria,pitch)
                inputs.append(input_parallel)
            with Pool(nb_processor) as p:
                results = p.starmap(func, inputs)
            for j in range(len(results)):
                best_score_this_pose = results[j][1]["optimization_score"]
                best_loc_this_pose = results[j][0]
                print("Score with rotation {} is {}".format(rotation_angle_options[j+rotation_index_this_processor],best_score_this_pose))
                if best_score_this_pose == -np.inf:
                    continue
                
                stone_ = inputs[j][2]
        
                # get the kinematics of the best position
            
                # add stone to wall
                stone_to_add = stone_
                best_loc_this_pose_pitch = [best_loc_this_pose[i]*pitch for i in range(3)]
                T = trimesh.transformations.translation_matrix(best_loc_this_pose_pitch)
                _ = stone_to_add.apply_transform(T)
                stone_image = convert_mesh_to_image(stone_to_add,wall.shape,pitch = pitch)
                if  best_score_this_pose> best_score:
                    best_rotate_pose_index = rotation_index_this_processor+j
                    best_score = best_score_this_pose
                    best_loc = results[j][0]
                    best_direction = results[j][1]["refinement_direction"]
                    
    return best_rotate_pose_index, best_score, best_loc,best_direction



def occ2points(coordinates):
    points  = []
    len = coordinates.shape[0]
    for i in range(len):
        points.append(np.array([round(coordinates[i,0]),round(coordinates[i,1]),round(coordinates[i,2])]))

    return np.array(points)

def generate_faces(points, pitch = 1):
    half_edge_size = 0.5*(1/pitch)
    corners = np.zeros((8*len(points),3))
    faces = np.zeros((6*len(points),4))
    for index in range(len(points)):
        corners[index*8]= np.array([points[index,0]-half_edge_size, points[index,1]-half_edge_size, points[index,2]-half_edge_size])
        corners[index*8+1]= np.array([points[index,0]+half_edge_size, points[index,1]-half_edge_size, points[index,2]-half_edge_size])
        corners[index*8+2]= np.array([points[index,0]-half_edge_size, points[index,1]+half_edge_size, points[index,2]-half_edge_size])
        corners[index*8+3]= np.array([points[index,0]+half_edge_size, points[index,1]+half_edge_size, points[index,2]-half_edge_size])
        corners[index*8+4]= np.array([points[index,0]-half_edge_size, points[index,1]-half_edge_size, points[index,2]+half_edge_size])
        corners[index*8+5]= np.array([points[index,0]+half_edge_size, points[index,1]-half_edge_size, points[index,2]+half_edge_size])
        corners[index*8+6]= np.array([points[index,0]-half_edge_size, points[index,1]+half_edge_size, points[index,2]+half_edge_size])
        corners[index*8+7]= np.array([points[index,0]+half_edge_size, points[index,1]+half_edge_size, points[index,2]+half_edge_size])
        base=len(points)+8*index
        faces[index*6]= np.array([base+2, base+3,base+1,base+0])
        faces[index*6+1]= np.array([base+4, base+5, base+7,base+6])
        faces[index*6+2]= np.array([base+3, base+2, base+6,base+7])
        faces[index*6+3]= np.array([base+0, base+1, base+5,base+4])
        faces[index*6+4]= np.array([base+2, base+0,base+4,base+6])
        faces[index*6+5]= np.array([base+1, base+3,base+7,base+5])
    
    return corners, faces

def write_ply(points, face_data, filename, text=True):

    points = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]

    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])

    face = np.empty(len(face_data),dtype=[('vertex_indices', 'i4', (4,))])
    face['vertex_indices'] = face_data

    ply_faces = PlyElement.describe(face, 'face')
    ply_vertexs = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([ply_vertexs, ply_faces], text=text).write(filename)

def writeocc(coordinates,save_path,filename,pitch = 1):
    points = occ2points(coordinates/pitch)
    #remove duplicate points
    points = np.unique(points,axis=0)

    #print(points.shape)
    corners, faces = generate_faces(points, pitch=pitch)
    if points.shape[0] == 0:
        print('the predicted mesh has zero point!')
    else:
        points = np.concatenate((points,corners),axis=0)
        write_ply(points, faces, os.path.join(save_path,filename))





def refine_placement_3d(stone_mesh_, wall_mesh,wall_size, direction = [1,1,1], threshold = 0.2,initial_step_size = 1,return_matrix=False,max_step_size=1):
    stone_mesh = stone_mesh_.copy()
    print("Refining placement in direction {}".format(direction))
    max_iteration = 3
    # set collision manager of the wall mesh
    wall_cm = trimesh.collision.CollisionManager()
    wall_cm.add_object('wall', wall_mesh)
    # cumulative transformation matrix
    x_cum = 0
    y_cum = 0
    z_cum = 0
    for i in range(max_iteration):
        #for dir_index in range(len(direction)):
        for dir_index in [0]:
            if direction[dir_index]==0:
                continue
            step_size = initial_step_size
            collision = False
            moved_distance = 0
            while step_size>threshold:
                #move stone mesh along the dir axis (in the order x,y,z direction) with a step size
                stone_mesh.vertices[:,dir_index] += step_size*int(direction[dir_index])
                moved_distance+=step_size*direction[dir_index]
                #check if the stone mesh is colliding with the wall mesh
                stone_cm = trimesh.collision.CollisionManager()
                stone_cm.add_object('stone', stone_mesh)
                collision = stone_cm.in_collision_other(wall_cm)
                #if collision, move the stone mesh back to the original position
                if collision:
                    stone_mesh.vertices[:,dir_index] -= step_size*int(direction[dir_index])
                    moved_distance-=step_size*direction[dir_index]
                    #print("Collision when moving in direction {} with step size {}".format(dir_index, step_size))
                    step_size-=threshold
                    continue
                #check if stone mesh is still inside wall size
                bbox_min = stone_mesh.bounds[0]
                if any(bbox_min)<0:
                    stone_mesh.vertices[:,dir_index] -= step_size*int(direction[dir_index])
                    moved_distance-=step_size*direction[dir_index]
                    #print("any(bbox_min)<0 when moving in direction {} with step size {}".format(dir_index, step_size))
                    step_size-=threshold
                    continue
                bbox_max = stone_mesh.bounds[1]
                if int(np.ceil(bbox_max[dir_index]))>wall_size[dir_index]-1 or int(np.floor(bbox_min[dir_index]))<1:
                    stone_mesh.vertices[:,dir_index] -= step_size*int(direction[dir_index])
                    moved_distance-=step_size*direction[dir_index]
                    #print("Out of bound when moving in direction {} with step size {}".format(dir_index, step_size))
                    step_size-=threshold
                    continue
                if step_size>=max_step_size:
                    break
            print("Moved distance is {} in direction {}".format(moved_distance,dir_index))
            if dir_index==0:
                x_cum+=moved_distance
            elif dir_index==1:
                y_cum+=moved_distance
            else:
                z_cum+=moved_distance
    if return_matrix:
        return stone_mesh,np.eye(4,4).dot([x_cum,y_cum,z_cum,1])
    return stone_mesh

def close_ground(ground_image):
    # dilate in positive direction
    index_array = np.arange(0,ground_image.shape[0]-1)
    #index_array= np.insert(index_array, 0, -1)
    for i in range(1,ground_image.shape[0]-1):
        ground_image[i:, :, :] = np.maximum(ground_image[0:-i, :, :], ground_image[i:, :, :])
    return ground_image


def save_voxel(bound_image_inner,pitch,result_dir,filename):
    coordinates = np.argwhere(bound_image_inner!=0)
    coordinates = coordinates*pitch
    writeocc(coordinates,result_dir,filename,\
            pitch = pitch)
def initialize_wall_voxel_with_bounds(bound_meshes,wall,wall_id_matrix,stone_index_matrix,result_dir,layer_i,pitch):
    global left_bound_pixel_value, right_bound_pixel_value
    wall_size = wall.shape
    # Set voxels under inner bound as 1
    inner_bound_mesh = bound_meshes[0].copy()
    #wall_mesh = trimesh.util.concatenate([wall_mesh,bound_mesh])
    bound_image_inner = convert_mesh_to_image(inner_bound_mesh,wall_size,pitch = pitch,add_minimum_ball=False)
    # save voxel
    save_voxel(bound_image_inner,pitch,result_dir,f'wall_{layer_i}_innerbound_voxel.ply')
    # set voxels above inner bound as 1
    inner_bound = np.zeros_like(wall)
    inner_bound[bound_image_inner!=0] = 1
    #inner_bound_filled = np.maximum.accumulate(inner_bound[::-1, :, :], axis=0)[::-1, :, :]
    inner_bound_filled = np.maximum.accumulate(inner_bound, axis=0)
    # update wall
    wall = np.where(inner_bound_filled==1,inner_bound_filled,wall)
    wall_id_matrix = np.where(bound_image_inner!=0,left_bound_pixel_value,wall_id_matrix)
    stone_index_matrix = np.where(bound_image_inner!=0,left_bound_pixel_value,stone_index_matrix)
    # save voxel
    save_voxel(inner_bound_filled,pitch,result_dir,f'wall_{layer_i}_innerbound_filled_voxel.ply')
    
    # Set voxels anove outer bound as 1
    outer_bound_mesh = bound_meshes[1].copy()
    #wall_mesh = trimesh.util.concatenate([wall_mesh,bound_mesh])
    bound_image_outer = convert_mesh_to_image(outer_bound_mesh,wall_size,pitch = pitch,add_minimum_ball=False)
    # save voxel
    save_voxel(bound_image_outer,pitch,result_dir,f'wall_{layer_i}_outerbound_voxel.ply')
    # set voxels anove outer bound as 1
    outer_bound = np.zeros_like(wall)
    outer_bound[bound_image_outer!=0] = 1
    #outer_bound_filled = np.maximum.accumulate(outer_bound, axis=0)
    outer_bound_filled = np.maximum.accumulate(outer_bound[::-1, :, :], axis=0)[::-1, :, :]
    # update wall
    wall = np.where(outer_bound_filled==1,outer_bound_filled,wall)
    wall_id_matrix = np.where(bound_image_outer!=0,right_bound_pixel_value,wall_id_matrix)
    stone_index_matrix = np.where(bound_image_outer!=0,right_bound_pixel_value,stone_index_matrix)
    # save voxel
    save_voxel(outer_bound_filled,pitch,result_dir,f'wall_{layer_i}_outerbound_filled_voxel.ply')
    # save wall voxel
    coordinates = np.argwhere(wall!=0)
    coordinates = coordinates*pitch
    writeocc(coordinates,result_dir,f'wall_{layer_i}_ground_bound_voxel.ply',\
            pitch = pitch)
    return wall, wall_id_matrix, stone_index_matrix

def place_stones_with_placed_stones(layer_i,nb_cand = 2,result_dir=None, sequence=None, stones=None, \
        placed_stones = None,wall_size=None,ground_mesh = None,bound_meshes = None,\
            rotation_angle_options = None,nb_processor = 1,pitch = 1,max_height = None):

    global left_bound_pixel_value, right_bound_pixel_value, base_pixel_value,global_result_dir
    global_result_dir = result_dir
    # initialize container for wall output
    logger.debug("initialize container for wall output")
    wall = np.zeros((wall_size[0], wall_size[1],wall_size[2]))
    wall_id_matrix = np.zeros((wall_size[0], wall_size[1],wall_size[2]))
    stone_index_matrix = np.zeros((wall_size[0], wall_size[1],wall_size[2]))
    edge_line_matrix_2 = np.zeros((wall_size[0], wall_size[1],wall_size[2]))
    edge_face_matrix_2 = np.zeros((wall_size[0], wall_size[1],wall_size[2]))
    edge_line_matrix_3 = np.zeros((wall_size[0], wall_size[1],wall_size[2]))
    edge_face_matrix_3 = np.zeros((wall_size[0], wall_size[1],wall_size[2]))
    interlocking_distance_2 = np.ones((wall_size[0], wall_size[1],wall_size[2]))*0.12
    interlocking_distance_3 = np.ones((wall_size[0], wall_size[1],wall_size[2]))*0.12
    # wall_mesh = trimesh.Trimesh()
    # read ground mesh as wall_mesh
    wall_mesh = ground_mesh.copy()
    # voxelize ground and boundary
    logger.debug("voxelize ground and boundary")
    #wall_mesh = trimesh.util.concatenate([wall_mesh,ground_mesh])
    ground_image = convert_mesh_to_image(ground_mesh,wall_size,pitch = pitch,add_minimum_ball=False)
    save_voxel(ground_image,pitch,result_dir,f'wall_{layer_i}_ground_voxel.ply')
    ground_image = close_ground(ground_image)
    save_voxel(ground_image,pitch,result_dir,f'wall_{layer_i}_ground_filled_voxel.ply')
    wall= np.where(ground_image!=0,ground_image,wall)
    wall_id_matrix = np.where(ground_image!=0,base_pixel_value,wall_id_matrix)
    stone_index_matrix = np.where(ground_image!=0,base_pixel_value,stone_index_matrix)

    wall, wall_id_matrix, stone_index_matrix=initialize_wall_voxel_with_bounds(bound_meshes,wall,wall_id_matrix,stone_index_matrix,result_dir,layer_i,pitch)
    # update wall_mesh
    wall_mesh = trimesh.util.concatenate([wall_mesh,bound_meshes[0]])
    wall_mesh = trimesh.util.concatenate([wall_mesh,bound_meshes[1]])
    # generate mask for max height
    if max_height is not None:
        wall[0:max_height] = 1
    #generate bound matrix for detecting joints on exteriour surfaces
    logger.debug("generate bound matrix for detecting joints on exteriour surfaces")
    wall_bound = np.where((wall_id_matrix==left_bound_pixel_value)|(wall_id_matrix==right_bound_pixel_value),1,0)
    kernel = np.ones((1,1,3),np.uint8)
    wall_bound_dilate_dim3 = scipy.ndimage.binary_dilation(wall_bound,structure=kernel).astype(wall_bound.dtype)
    #voxelize placed stones
    logger.debug("voxelize placed stones")

    save_voxel(wall_id_matrix,pitch,result_dir,f'wall_{layer_i}_ground_add_stones_all_voxel.ply')
    # check if there is any zero pixel
    if len(np.argwhere(edge_line_matrix_2!=0))==0:
        interlocking_distance_2 = np.ones_like(edge_line_matrix_2)*0.12
    else:
        # compute the distance to the boundary
        interlocking_distance_2 = scipy.ndimage.distance_transform_edt(np.where(edge_line_matrix_2==0,1,0), return_distances=True, return_indices=False)
    if len(np.argwhere(edge_line_matrix_3!=0))==0:
        interlocking_distance_3 = np.ones_like(edge_line_matrix_3)*0.12
    else:
        # compute the distance to the boundary
        interlocking_distance_3 = scipy.ndimage.distance_transform_edt(np.where(edge_line_matrix_3==0,1,0), return_distances=True, return_indices=False)
    #summarize all matrix
    wall_other_matrixs = {}
    wall_other_matrixs['interlocking_distance_2'] = interlocking_distance_2
    wall_other_matrixs['interlocking_distance_3'] = interlocking_distance_3
    wall_other_matrixs['edge_face_matrix_2'] = edge_face_matrix_2
    wall_other_matrixs['edge_face_matrix_3'] = edge_face_matrix_3
    wall_other_matrixs['wall_seg_matrix'] = wall_id_matrix
    wall_other_matrixs['stone_index_matrix'] = stone_index_matrix

    # recycle unplaced stones
    unplaced_stones = dict()
    # start iterating the stone sequence
    step_i = 0
    available_stones_id = sequence.copy()
    consecutive_fail = 0
    while consecutive_fail<=10:
        print(f"------------------ Step {step_i} ------------------")
        print(f"Number of available stones: {len(available_stones_id)}")
        logger.debug(f"------------------ Step {step_i} ------------------")
        logger.debug(f"Number of available stones: {len(available_stones_id)}")
        current_rotation_angle_options = rotation_angle_options
        #get candidate stones id
        if len(available_stones_id)>nb_cand:
            candidate_stones_id = available_stones_id[:nb_cand]
        else:
            candidate_stones_id = available_stones_id.copy()
        optimization_results = []
        for single_candidate_id in candidate_stones_id:
            logger.debug(f"stone id {single_candidate_id}")
            # get the stone mesh
            stone_mesh = stones[single_candidate_id]['mesh']
            # optimize stone placement and rotation
            nb_processor = min(nb_processor, len(rotation_angle_options))
            best_rotate_pose_index, best_score, best_loc,refine_direction = get_best_placement(wall, \
                wall_other_matrixs,stone_mesh,current_rotation_angle_options,\
                    weight_height = 1,func = add_stone_3d,\
                        rotation_function = rotate_face_up,nb_processor = nb_processor,\
                            relaxed_mason_criteria = False,result_dir = result_dir,\
                                iteration = step_i,pitch = pitch,wall_mesh = wall_mesh,kinematic_check = False)
            optimization_results.append({"id":single_candidate_id,"best_rotate_pose_index":best_rotate_pose_index,\
                                         "best_score":best_score,"best_loc":best_loc,"refine_direction":refine_direction})
        
        #compare candidates
        best_cand_score = -np.inf
        best_cand_index = -1
        remove_cand_ids = []
        for cand_index_in_cand_list in range(len(optimization_results)):
            if optimization_results[cand_index_in_cand_list]['best_score']<=-np.inf:
                remove_cand_ids.append(optimization_results[cand_index_in_cand_list]['id'])
            elif optimization_results[cand_index_in_cand_list]['best_score']>best_cand_score:
                best_cand_score = optimization_results[cand_index_in_cand_list]['best_score']
                best_cand_index = cand_index_in_cand_list
            else:
                #not the best, not infeasible
                pass
        for remove_cand_id in remove_cand_ids:
            print(f"Stone {remove_cand_id} is move to end in availble set.")
            available_stones_id.remove(remove_cand_id)
            available_stones_id.append(remove_cand_id)
            #write the stone mesh
            stone_mesh = stones[remove_cand_id]['mesh']
            stone_mesh.export(result_dir+f'/wall_{layer_i}_iteration{step_i}_invalid_stone_{remove_cand_id}_best_pose_random_sequence.ply')

        if best_cand_index == -1:
            print("No feasible placement")
            consecutive_fail+=1
            # add stone to unplaced stones
        else:
            consecutive_fail = 0
            available_stones_id.remove(optimization_results[best_cand_index]['id'])
            # add stone to wall
            stone_index = optimization_results[best_cand_index]['id']
            best_rotate_pose_index = optimization_results[best_cand_index]['best_rotate_pose_index']
            best_loc = optimization_results[best_cand_index]['best_loc']
            refine_direction = optimization_results[best_cand_index]['refine_direction']

            stone_to_add = stones[stone_index]['mesh']
            stone_mesh = rotate_face_up(stone_to_add,current_rotation_angle_options[best_rotate_pose_index],return_matrix = True)
            # write Rstar to file
            #np.savetxt(result_dir+f'/wall_{wall_i}_iteration{step_i}_valid_Rstar_{stone_index}_best_pose_random_sequence.txt',Rstar)
            best_loc = [best_loc[i]*pitch for i in range(3)]
            T = trimesh.transformations.translation_matrix(best_loc)
            print("Best location is ",best_loc)
            # write T to file
            #np.savetxt(result_dir+f'/wall_{wall_i}_iteration{step_i}_valid_Tstar_{stone_index}_best_pose_random_sequence.txt',T)
            _ = stone_mesh.apply_transform(T)
            # write mesh to file
            stone_mesh.export(result_dir+f'/wall_{layer_i}_iteration{step_i}_valid_stone_{stone_index}_best_pose_random_sequence.ply')
            start_timer = time.time()
            # continous refinement
            wall_size_xyz = [wall_size[i]*pitch for i in range(3)]
            refined_mesh,Tr = refine_placement_3d(stone_mesh, wall_mesh,wall_size_xyz,direction=refine_direction, \
                                                  threshold = pitch/5,initial_step_size = pitch,return_matrix=True,\
                                                    max_step_size = pitch)                        
            out_mesh_file_type = 'obj'
            save_filename =result_dir+f'/wall_{layer_i}_iteration{step_i}_valid_refined_{stone_index}_best_pose_random_sequence.'+out_mesh_file_type 
            refined_mesh.export(save_filename,file_type=out_mesh_file_type, include_texture=False,\
                                mtl_name = f'iteration{step_i}_stone{stone_index}.mtl')

            # save voxel
            stone_image = convert_mesh_to_image(refined_mesh,wall_size,pitch = pitch)
            coordinates = np.argwhere(stone_image!=0)
            coordinates = coordinates*pitch
            writeocc(coordinates,result_dir,f'wall_{layer_i}_iteration{step_i}_valid_voxel_{stone_index}_best_pose_random_sequence.ply',\
                    pitch = pitch)
            # merge wall mesh and stone mesh
            wall_mesh = trimesh.util.concatenate([wall_mesh,refined_mesh])
            #update wall
            wall= np.where(stone_image!=0,stone_image,wall)#overlapping exists between matrix because of refined mesh position
            wall_id_matrix = np.where(stone_image!=0,(step_i+1)*stone_image,wall_id_matrix)
            stone_index_matrix = np.where(stone_image!=0,(stone_index+1)*stone_image,stone_index_matrix)
            wall_other_matrixs['wall_seg_matrix'] = wall_id_matrix
            wall_other_matrixs['stone_index_matrix'] = stone_index_matrix

        step_i+=1
        if not available_stones_id:
            break
    print("Assembly finish!")

