import copy
from tqdm import tqdm
import glm
import numpy as np
import pickle
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import smplx
import torch

import math

mano_layer = {}
root_right = None
root_left = None
seq_dict = {}


# get mesh from MANO model
def compute_mano_hands(mano_param, hand_type, layer):
    mano_pose = torch.FloatTensor(mano_param['pose']).view(-1, 3)
    root_pose = mano_pose[0].view(1, 3)
    hand_pose = mano_pose[1:, :].view(1, -1)
    shape = torch.FloatTensor(mano_param['shape']).view(1, -1)
    trans = torch.FloatTensor(mano_param['trans']).view(1, -1)
    output = layer[hand_type](global_orient=root_pose, hand_pose=hand_pose, betas=shape, transl=trans)
    mesh = output.vertices[0].numpy()  # meter unit
    joints = output.joints[0].numpy()  # meter unit
    faces = layer[hand_type].faces

    return mesh, faces, joints


# initialize mano models
def init_mano(file_name):
    global mano_layer
    global seq_dict

    mano_layer = None
    seq_dict = None

    with open(file_name, 'rb') as f:
        seq_dict = pickle.load(f)

    mano_layer = {'right': smplx.create(model_path='model', model_type='mano', use_pca=False, is_rhand=True,
                                        flat_hand_mean=True),
                  'left': smplx.create(model_path='model', model_type='mano', use_pca=False, is_rhand=False,
                                       flat_hand_mean=True)}

    return len(seq_dict)


# interpolate to desired frame rate
def interpolate_sequence(fps_input, fps_output):
    global seq_dict

    length_in = len(seq_dict)
    x_in = np.linspace(0, length_in, num=length_in, endpoint=False)

    poses_right_in = np.zeros((length_in, 48))
    shapes_right_in = np.zeros((length_in, 10))
    trans_right_in = np.zeros((length_in, 3))
    poses_left_in = np.zeros((length_in, 48))
    shapes_left_in = np.zeros((length_in, 10))
    trans_left_in = np.zeros((length_in, 3))

    for f in seq_dict.keys():
        right_hand = seq_dict[f][0]
        left_hand = seq_dict[f][1]

        poses_right_in[f, :] = right_hand['pose']
        poses_left_in[f, :] = left_hand['pose']
        shapes_right_in[f, :] = right_hand['shape']
        shapes_left_in[f, :] = left_hand['shape']
        trans_right_in[f, :] = right_hand['trans']
        trans_left_in[f, :] = left_hand['trans']

    slerp_poses_right = []
    slerp_poses_left = []
    interps_shapes_right = []
    interps_trans_right = []
    interps_shapes_left = []
    interps_trans_left = []

    for i in range(16):
        y_right = R.from_rotvec(poses_right_in[:, 3 * i: 3 * (i + 1)])
        y_left = R.from_rotvec(poses_left_in[:, 3 * i: 3 * (i + 1)])
        slerp_poses_right.append(Slerp(x_in, y_right))
        slerp_poses_left.append(Slerp(x_in, y_left))

    for i in range(shapes_right_in.shape[1]):
        y_right = shapes_right_in[:, i]
        y_left = shapes_left_in[:, i]
        interps_shapes_right.append(interp1d(x_in, y_right, kind='cubic'))
        interps_shapes_left.append(interp1d(x_in, y_left, kind='cubic'))

    for i in range(trans_right_in.shape[1]):
        y_right = trans_right_in[:, i]
        y_left = trans_left_in[:, i]
        interps_trans_right.append(interp1d(x_in, y_right, kind='cubic'))
        interps_trans_left.append(interp1d(x_in, y_left, kind='cubic'))

    rate = fps_output / fps_input
    length_out = int(round(x_in.shape[0] * rate))
    x_out = np.linspace(0, np.max(x_in), num=length_out, endpoint=True)

    poses_right_out = np.zeros((length_out, 48))
    shapes_right_out = np.zeros((length_out, 10))
    trans_right_out = np.zeros((length_out, 3))
    poses_left_out = np.zeros((length_out, 48))
    shapes_left_out = np.zeros((length_out, 10))
    trans_left_out = np.zeros((length_out, 3))

    for s, slerp in enumerate(slerp_poses_right):
        poses_right_out[:, 3 * s: 3 * (s + 1)] = slerp(x_out).as_rotvec()

    for s, slerp in enumerate(slerp_poses_left):
        poses_left_out[:, 3 * s: 3 * (s + 1)] = slerp(x_out).as_rotvec()

    for i, interp in enumerate(interps_shapes_right):
        shapes_right_out[:, i] = interp(x_out)

    for i, interp in enumerate(interps_shapes_left):
        shapes_left_out[:, i] = interp(x_out)

    for i, interp in enumerate(interps_trans_right):
        trans_right_out[:, i] = interp(x_out)

    for i, interp in enumerate(interps_trans_left):
        trans_left_out[:, i] = interp(x_out)

    seq_dict_out = {}

    for f in range(length_out):
        seq_dict_out[f] = [{'pose': poses_right_out[f, :], 'shape': shapes_right_out[f, :],
                            'trans': trans_right_out[f, :], 'hand_type': 'right'},
                           {'pose': poses_left_out[f, :], 'shape': shapes_left_out[f, :], 'trans': trans_left_out[f, :],
                            'hand_type': 'left'}]

    seq_dict = seq_dict_out

    # save to a file for later
    with open('sequences/1000fps/raw_sequence0.pkl', 'wb') as f:
        pickle.dump(seq_dict, f)


# transforms global MANO translations and rotations into the OpenGL camera coordinate system
def transform_coordinate_system(s):
    global seq_dict

    hands_avg_all = [np.array([1.15030333, -0.26061168, 0.78577989]), np.array([1.12174921, -0.20740417, 0.81597976]),
                     np.array([1.16503733, -0.31407652, 0.81827346]), np.array([1.03878048, -0.27550721, 0.82758726]),
                     np.array([1.03542053, -0.1853186, 0.77306902]), np.array([0.98415266, -0.42346881, 0.76287726]),
                     np.array([0.99070947, -0.40857825, 0.75110521]), np.array([1.00070618, -0.40154313, 0.77840039])]
    hands_avg = hands_avg_all[s]

    far = 1.0
    camera_relative = glm.vec3(0.5 * far, 0.0, 0.0)
    forward = glm.vec3(-1.0, 0.0, 0.0)
    up = glm.vec3(0.0, 0.0, 1.0)

    view_matrix = np.array(glm.lookAt(glm.vec3(hands_avg) + camera_relative,
                                      glm.vec3(hands_avg) + camera_relative + forward,
                                      up))
    rot_vm = view_matrix[:3, :3]
    t_cam = hands_avg + camera_relative

    seq_dict_out = {}

    for f in tqdm(seq_dict.keys()):
        seq_dict_out[f] = []

        for h, hand in enumerate(seq_dict[f]):
            rot_mano = hand['pose'][:3]
            trans_mano = hand['trans']
            trans_manocam = trans_mano - t_cam
            rot_new = R.from_matrix(rot_vm) * R.from_rotvec(rot_mano)
            hand['pose'][:3] = rot_new.as_rotvec()

            base = {'pose': hand['pose'], 'trans': np.zeros(3), 'shape': hand['shape']}

            _, _, joints = compute_mano_hands(base, hand['hand_type'], mano_layer)
            root = joints[0, ...]

            trans_new = -root + rot_vm.dot(root + trans_manocam)

            mano_param_updated = copy.deepcopy(hand)
            mano_param_updated['pose'][:3] = hand['pose'][:3]
            mano_param_updated['trans'] = trans_new

            seq_dict_out[f].append({'pose': mano_param_updated['pose'], 'shape': mano_param_updated['shape'],
                                    'trans': mano_param_updated['trans'], 'hand_type': 'right' if h == 0 else 'left'})

    seq_dict = seq_dict_out

    # save to a file for later
    with open('sequences/raw_sequence' + str(s) + '.pkl', 'wb') as f:
        pickle.dump(seq_dict, f)


# returns two hands for a given frame in the recorded (interpolated) sequence
def get_mano_hands(frame, angle_augmentation, angle_position):
    global mano_layer
    global root_left
    global root_right
    global seq_dict

    params = seq_dict[frame]

    camera_relative = glm.vec3(0.0, 0.0, 0.0)
    hands_avg = glm.vec3(0.0, 0.0, -0.5)
    forward = glm.vec3(0.0, 0.0, -1.0)

    view_matrix = np.eye(4)

    if angle_position is not None:
        rot_aa = np.array([[math.cos(angle_augmentation), 0.0, math.sin(angle_augmentation)],
                           [0.0, 1.0, 0.0],
                           [-math.sin(angle_augmentation), 0.0, math.cos(angle_augmentation)]])
        rot_ap = np.array([[math.cos(angle_position), -math.sin(angle_position), 0.0],
                           [math.sin(angle_position), math.cos(angle_position), 0.0],
                           [0.0, 0.0, 1.0]])

        camera_relative_transformed = rot_aa.dot(camera_relative - hands_avg) + hands_avg
        forward_transformed = rot_aa.dot(forward)
        camera_relative = rot_ap.dot(camera_relative_transformed)
        forward = rot_ap.dot(forward_transformed)
        line_target_2d = np.array([-forward[2], forward[0]])
        line_target_2d /= np.linalg.norm(line_target_2d)
        right_horizontal = glm.vec3(line_target_2d[0], 0.0, line_target_2d[1])
        up = np.cross(-forward, right_horizontal)

        view_matrix = np.zeros((4, 4))
        view_matrix[0, :3] = np.array(right_horizontal)
        view_matrix[1, :3] = np.array(up)
        view_matrix[2, :3] = np.array(-forward)
        view_matrix[:3, 3] = -view_matrix[:3, :3].dot(hands_avg + camera_relative)
        view_matrix[3, 3] = 1.0

    rot_vm = view_matrix[:3, :3]

    if root_left is None or root_right is None:
        # assume shape to be zero
        base = {'pose': np.zeros(48), 'trans': np.zeros(3), 'shape': np.zeros(10)}

        _, _, joints_left = compute_mano_hands(base, 'left', mano_layer)
        _, _, joints_right = compute_mano_hands(base, 'right', mano_layer)
        root_left = joints_left[0, ...]
        root_right = joints_right[0, ...]

    mano_hands = []

    for param_id in range(len(params)):
        param = params[param_id]

        rot_mano = param['pose'][:3]
        trans_mano = param['trans']
        trans_manocam = trans_mano - camera_relative
        rot_new = R.from_matrix(rot_vm) * R.from_rotvec(rot_mano)
        param['pose'][:3] = rot_new.as_rotvec()

        hand_type = param['hand_type']
        root = root_left if hand_type == 'left' else root_right

        trans_new = -root + rot_vm.dot(root + trans_manocam)

        param_updated = copy.deepcopy(param)
        param_updated['pose'][:3] = param['pose'][:3]
        param_updated['trans'] = trans_new

        vertices, faces, mano_joints = compute_mano_hands(param_updated, hand_type, mano_layer)
        mano_hands.append((vertices, faces, mano_joints))

    return mano_hands
