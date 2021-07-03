import math
import numpy as np
import pickle
from scipy.spatial.transform import Rotation as R
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


# interpolate to desired frame rate (currently linear)
def interpolate_sequence(fps, fps_target):
    global seq_dict

    num_frames = int(round(len(seq_dict) * fps_target / fps))
    step = (fps - 1) / (fps_target - 1)
    seq_dict_new = {0: seq_dict[0]}

    for f in range(1, num_frames):
        print(f)
        seq_dict_new[f] = [{}, {}]

        before = int(math.floor(f * step))
        after = int(math.ceil(f * step))
        factor_after = f * step - before
        factor_before = 1 - factor_after

        pose_0_before = seq_dict[before][0]['pose']
        pose_0_after = seq_dict[after][0]['pose']

        shape_0_before = seq_dict[before][0]['shape']
        shape_0_after = seq_dict[after][0]['shape']

        trans_0_before = seq_dict[before][0]['trans']
        trans_0_after = seq_dict[after][0]['trans']

        hand_type_0 = seq_dict[before][0]['hand_type']

        pose_1_before = seq_dict[before][1]['pose']
        pose_1_after = seq_dict[after][1]['pose']

        shape_1_before = seq_dict[before][1]['shape']
        shape_1_after = seq_dict[after][1]['shape']

        trans_1_before = seq_dict[before][1]['trans']
        trans_1_after = seq_dict[after][1]['trans']

        hand_type_1 = seq_dict[before][1]['hand_type']

        pose_0_new = np.zeros(48)

        for i in range(16):
            r_before = R.from_rotvec(pose_0_before[3 * i:3 * (i + 1)])
            r_after = R.from_rotvec(pose_0_after[3 * i:3 * (i + 1)])
            q_before = r_before.as_quat()
            q_after = r_after.as_quat()
            q_interpolated = factor_before * q_before + factor_after * q_after
            rotvec_interpolated = R.from_quat(q_interpolated).as_rotvec()
            pose_0_new[3 * i:3 * (i + 1)] = rotvec_interpolated

        seq_dict_new[f][0]['pose'] = pose_0_new
        seq_dict_new[f][0]['shape'] = factor_before * shape_0_before + factor_after * shape_0_after
        seq_dict_new[f][0]['trans'] = factor_before * trans_0_before + factor_after * trans_0_after
        seq_dict_new[f][0]['hand_type'] = hand_type_0

        pose_1_new = np.zeros(48)

        for i in range(16):
            r_before = R.from_rotvec(pose_1_before[3 * i:3 * (i + 1)])
            r_after = R.from_rotvec(pose_1_after[3 * i:3 * (i + 1)])
            q_before = r_before.as_quat()
            q_after = r_after.as_quat()
            q_interpolated = factor_before * q_before + factor_after * q_after
            rotvec_interpolated = R.from_quat(q_interpolated).as_rotvec()
            pose_1_new[3 * i:3 * (i + 1)] = rotvec_interpolated

        seq_dict_new[f][1]['pose'] = pose_1_new
        seq_dict_new[f][1]['shape'] = factor_before * shape_1_before + factor_after * shape_1_after
        seq_dict_new[f][1]['trans'] = factor_before * trans_1_before + factor_after * trans_1_after
        seq_dict_new[f][1]['hand_type'] = hand_type_1

    seq_dict = seq_dict_new

    # save to a file for later
    with open('sequences/raw_sequence_1000fps.pkl', 'wb') as f:
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
