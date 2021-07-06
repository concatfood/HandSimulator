import numpy as np
import pickle
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import smplx
import torch

mano_layer = {}
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
        seq_dict_out[f] = [{'pose': poses_right_out[f, :], 'shape': shapes_right_out[f, :], 'trans': trans_right_out[f, :],
                            'hand_type': 'right'},
                           {'pose': poses_left_out[f, :], 'shape': shapes_left_out[f, :], 'trans': trans_left_out[f, :],
                            'hand_type': 'left'}]

    seq_dict = seq_dict_out

    # save to a file for later
    with open('sequences/1000fps/raw_sequence0.pkl', 'wb') as f:
        pickle.dump(seq_dict, f)


# returns two hands for a given frame in the recorded (interpolated) sequence
def get_mano_hands(frame):
    global mano_layer
    global seq_dict

    params = seq_dict[frame]

    mano_hands = []

    for param_id in range(len(params)):
        param = params[param_id]
        hand_type = param['hand_type']
        vertices, faces, mano_joints = compute_mano_hands(param, hand_type, mano_layer)
        mano_hands.append((vertices, faces, mano_joints))

    return mano_hands


# removes tracking error caused by the gloves
# an error is defined by the continuation of two consecutive frames with the same tracking values
# this step is not used anymore
def remove_tracking_errors(threshold=1e-9):
    global seq_dict

    length_in = len(seq_dict)
    x_in = np.linspace(0, length_in, num=length_in, endpoint=False)

    poses_right_in = np.zeros((length_in, 48))
    shapes_right_in = np.zeros((length_in, 10))
    trans_right_in = np.zeros((length_in, 3))
    poses_left_in = np.zeros((length_in, 48))
    shapes_left_in = np.zeros((length_in, 10))
    trans_left_in = np.zeros((length_in, 3))

    to_remove = []

    for t in range(0, length_in - 1):
        params_right = np.concatenate((poses_right_in[t:t + 2, :], shapes_right_in[t:t + 2, :],
                                       trans_right_in[t:t + 2, :]), axis=1)
        params_left = np.concatenate((poses_left_in[t:t + 2, :], shapes_left_in[t:t + 2, :],
                                      trans_left_in[t:t + 2, :]), axis=1)

        diff_right = params_right[1, :] - params_right[0, :]
        diff_left = params_left[1, :] - params_left[0, :]

        if abs(np.mean(diff_right)) < threshold or abs(np.mean(diff_left)) < threshold:
            to_remove.append(t)
            to_remove.append(t + 1)

    x_in = np.delete(x_in, to_remove)
    poses_right_in = np.delete(poses_right_in, to_remove, axis=0)
    shapes_right_in = np.delete(shapes_right_in, to_remove, axis=0)
    trans_right_in = np.delete(trans_right_in, to_remove, axis=0)
    poses_left_in = np.delete(poses_left_in, to_remove, axis=0)
    shapes_left_in = np.delete(shapes_left_in, to_remove, axis=0)
    trans_left_in = np.delete(trans_left_in, to_remove, axis=0)

    # ...
