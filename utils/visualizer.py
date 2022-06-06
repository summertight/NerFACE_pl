import numpy as np
import torchvision
from scipy.spatial.transform import Rotation as R
import math
import json as js
import random

def cast_to_image(tensor):
    img = tensor.clamp(0.0, 1.0).detach().cpu().numpy()
    return img # (H, W, 3) or (H, W)


def explicit_pose_control(pose_matrix, angle):
    """
        Input:
            rot_matrix: Original Rotation Matrix
            pose_matrix: Angle to change for each axis (Radian)
        Output:
            pose_matrix_list: yaw, pitch, roll을 각각 angle만큼 변경한 rotation matrix의 List
    """
    
    trans= pose_matrix[:, -1]
    rot_matrix = pose_matrix[..., :-1]
    
    radian = (angle / 180) * math.pi
    rot_vec_org = R.from_matrix(rot_matrix).as_rotvec()
    
    pose_matrix_list = []
    pose_matrix_list.append(pose_matrix)
    
    for i in range(3):
        rot_vec_pos = rot_vec_org.copy()
        rot_vec_neg = rot_vec_org.copy()
        
        rot_vec_pos[i] += radian
        rot_vec_neg[i] -= radian

        rot_mat_pos = np.hstack((R.from_rotvec(rot_vec_pos).as_matrix(), trans[:, None]))
        rot_mat_neg = np.hstack((R.from_rotvec(rot_vec_neg).as_matrix(), trans[:, None]))

        pose_matrix_list += [rot_mat_pos, rot_mat_neg]

    return pose_matrix_list

def explicit_expr_control(expr_vector, ctrl_num):
    
    minmax_info_path = 'NerFACE_pl/3dmm_expr_minmax.json'
    
    with open(minmax_info_path, "r") as json_file:
        minmax_data = js.load(json_file)
    
    idx_list = random.sample(range(77), ctrl_num)
    expr_list = []
    neutral_expr = np.zeros_like(expr_vector)
    
    
    for i in range(ctrl_num):
        expr_p = neutral_expr.copy()
        expr_n = neutral_expr.copy()
        
        ctrl_loc = idx_list[i]
        
        expr_p[ctrl_loc] = minmax_data['max'][ctrl_loc]
        expr_n[ctrl_loc] = minmax_data['min'][ctrl_loc]
        
        expr_list += [expr_p, expr_n]
    
    return expr_list
    


if __name__ == "__main__":
    #rot_matrix = np.ones((3, 4))
    #angle = 30
    #pose_matrix_list = rotation_yaw_perturb(rot_matrix, angle)
    pass