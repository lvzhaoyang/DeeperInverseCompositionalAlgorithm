"""
Data loader for TUM RGBD benchmark

@author: Zhaoyang Lv
@date: March 2019
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys, os, random
import pickle

import numpy as np
import os.path as osp
import torch.utils.data as data

from scipy.misc import imread
from tqdm import tqdm
from transforms3d import quaternions
from cv2 import resize, INTER_NEAREST

"""
The following scripts use the directory structure as:

root
---- fr1 
-------- rgbd_dataset_freiburg1_360
-------- rgbd_dataset_freiburg1_desk 
-------- rgbd_dataset_freiburg1_desk2 
-------- ...
---- fr2 
-------- rgbd_dataset_freiburg2_360_hemisphere
-------- rgbd_dataset_freiburg2_360_kidnap
-------- rgbd_dataset_freiburg2_coke
-------- ....
---- fr3
-------- rgbd_dataset_freiburg3_cabinet
-------- rgbd_dataset_freiburg3_long_office_household
-------- rgbd_dataset_freiburg3_nostructure_notexture_far
-------- .... 
"""

def tum_trainval_dict(): 
    """ the sequence dictionary of TUM dataset
        https://vision.in.tum.de/data/datasets/rgbd-dataset/download

        The calibration parameters refers to: 
        https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats 
    """
    return  {
        'fr1': { 
            'calib': [525.0, 525.0, 319.5, 239.5],
            'seq': ['rgbd_dataset_freiburg1_desk2',
                'rgbd_dataset_freiburg1_floor',
                'rgbd_dataset_freiburg1_room',
                'rgbd_dataset_freiburg1_xyz',
                'rgbd_dataset_freiburg1_rpy',
                'rgbd_dataset_freiburg1_plant',
                'rgbd_dataset_freiburg1_teddy']
        },

        'fr2': {
            'calib': [525.0, 525.0, 319.5, 239.5],
            'seq': ['rgbd_dataset_freiburg2_360_hemisphere',
                'rgbd_dataset_freiburg2_large_no_loop',
                'rgbd_dataset_freiburg2_large_with_loop',
                'rgbd_dataset_freiburg2_pioneer_slam',
                'rgbd_dataset_freiburg2_pioneer_slam2',
                'rgbd_dataset_freiburg2_pioneer_slam3',
                'rgbd_dataset_freiburg2_xyz',
                'rgbd_dataset_freiburg2_360_kidnap',
                'rgbd_dataset_freiburg2_rpy',
                'rgbd_dataset_freiburg2_coke', 
                'rgbd_dataset_freiburg2_desk_with_person',
                'rgbd_dataset_freiburg2_dishes',
                'rgbd_dataset_freiburg2_flowerbouquet_brownbackground',
                'rgbd_dataset_freiburg2_metallic_sphere2',
                'rgbd_dataset_freiburg2_flowerbouquet'
                ]
        }, 

        'fr3': {
            'calib': [525.0, 525.0, 319.5, 239.5],
            'seq': [
                'rgbd_dataset_freiburg3_walking_halfsphere',
                'rgbd_dataset_freiburg3_walking_rpy',
                'rgbd_dataset_freiburg3_cabinet',  
                'rgbd_dataset_freiburg3_nostructure_notexture_far', 
                'rgbd_dataset_freiburg3_nostructure_notexture_near_withloop', 
                'rgbd_dataset_freiburg3_nostructure_texture_far', 
                'rgbd_dataset_freiburg3_nostructure_texture_near_withloop', 
                'rgbd_dataset_freiburg3_sitting_rpy', 
                'rgbd_dataset_freiburg3_sitting_static', 
                'rgbd_dataset_freiburg3_sitting_xyz', 
                'rgbd_dataset_freiburg3_structure_notexture_near', 
                'rgbd_dataset_freiburg3_structure_texture_far', 
                'rgbd_dataset_freiburg3_structure_texture_near',
                'rgbd_dataset_freiburg3_teddy']
        }
    }

def tum_test_dict():
    """ the trajectorys held out for testing TUM dataset
    """
    return  {
        'fr1': { 
            'calib': [525.0, 525.0, 319.5, 239.5],
            'seq': ['rgbd_dataset_freiburg1_360',
                'rgbd_dataset_freiburg1_desk'] 
        },

        'fr2': {
            'calib': [525.0, 525.0, 319.5, 239.5],
            'seq': ['rgbd_dataset_freiburg2_desk',
                'rgbd_dataset_freiburg2_pioneer_360']
        }, 

        'fr3': {
            'calib': [525.0, 525.0, 319.5, 239.5],
            'seq': ['rgbd_dataset_freiburg3_walking_static', # dynamic scene
                'rgbd_dataset_freiburg3_walking_xyz',        # dynamic scene
                'rgbd_dataset_freiburg3_long_office_household']
        }
    }

class TUM(data.Dataset): 

    base = 'data'

    def __init__(self, root = '', category='train', 
        keyframes=[1], data_transform=None, select_traj=None):
        """
        :param the root directory of the data 
        :param select the category (train, validation,test)
        :param select the number of keyframes 
            Test data only support one keyfame at one time
            Train/validation data support mixing different keyframes 
        :param select one particular trajectory at runtime 
            Only support for testing 
        """ 
        super(TUM, self).__init__()
        
        self.image_seq = []
        self.depth_seq = []
        self.invalid_seq = []
        self.cam_pose_seq= []
        self.calib = []
        self.seq_names = []

        self.ids = 0
        self.seq_acc_ids = [0]
        self.keyframes = keyframes

        self.transforms = data_transform

        if category == 'test':
            self.__load_test(root+'/data_tum', select_traj)
        else: # train and validation
            self.__load_train_val(root+'/data_tum', category)

        # downscale the input image to a quarter
        self.fx_s = 0.25
        self.fy_s = 0.25

        print('TUM dataloader for {:} using keyframe {:}: \
            {:} valid frames'.format(category, keyframes, self.ids))

    def __load_train_val(self, root, category):

        tum_data = tum_trainval_dict()

        for ks, scene in tum_data.items():
            for seq_name in scene['seq']: 

                seq_path = osp.join(ks, seq_name)
                self.calib.append(scene['calib'])
                # synchronized trajectory file 
                sync_traj_file = osp.join(root, seq_path, 'sync_trajectory.pkl') 
                if not osp.isfile(sync_traj_file):
                    print("The synchronized trajectory file {:} has not been generated.".format(seq_path))
                    print("Generate it now...")
                    write_sync_trajectory(root, ks, seq_name)

                with open(sync_traj_file, 'rb') as p:
                    trainval = pickle.load(p)
                    total_num = len(trainval)
                    # the ratio to split the train & validation set                        
                    if category == 'train': 
                        start_idx, end_idx = 0, int(0.95*total_num)
                    else:
                        start_idx, end_idx = int(0.95*total_num), total_num

                    images = [trainval[idx][1] for idx in range(start_idx, end_idx)]
                    depths = [trainval[idx][2] for idx in range(start_idx, end_idx)]
                    extrin = [tq2mat(trainval[idx][0]) for idx in range(start_idx, end_idx)]
                    self.image_seq.append(images)
                    self.depth_seq.append(depths)
                    self.cam_pose_seq.append(extrin)
                    self.seq_names.append(seq_path)
                    self.ids += max(0, len(images) - max(self.keyframes))
                    self.seq_acc_ids.append(self.ids)

    def __load_test(self, root, select_traj=None):
        """ Note: 
        The test trajectory is loaded slightly different from the train/validation trajectory. 
        We only select keyframes from the entire trajectory, rather than use every individual frame. 
        For a given trajectory of length N, using key-frame 2, the train/validation set will use 
        [[1, 3], [2, 4], [3, 5],...[N-1, N]], 
        while test set will use pair 
        [[1, 3], [3, 5], [5, 7],...[N-1, N]]
        This difference result in a change in the trajectory length when using different keyframes.

        The benefit of sampling keyframes of the test set is that the output is a more reasonable trajectory;
        And in training/validation, we fully leverage every pair of image.
        """

        tum_data = tum_test_dict()

        assert(len(self.keyframes) == 1) 
        kf = self.keyframes[0]
        self.keyframes = [1]

        for ks, scene in tum_data.items():
            for seq_name in scene['seq']: 
                seq_path = osp.join(ks, seq_name)

                if select_traj is not None: 
                    if seq_path != select_traj: continue

                self.calib.append(scene['calib'])
                # synchronized trajectory file 
                sync_traj_file = osp.join(root, seq_path, 'sync_trajectory.pkl') 
                if not osp.isfile(sync_traj_file):
                    print("The synchronized trajectory file {:} has not been generated.".format(seq_path))
                    print("Generate it now...")
                    write_sync_trajectory(root, ks, seq_name)

                with open(sync_traj_file, 'rb') as p:
                    frames = pickle.load(p)
                    total_num = len(frames)

                    images = [frames[idx][1] for idx in range(0, total_num, kf)]
                    depths = [frames[idx][2] for idx in range(0, total_num, kf)]
                    extrin = [tq2mat(frames[idx][0]) for idx in range(0, total_num, kf)]
                    self.image_seq.append(images)
                    self.depth_seq.append(depths)
                    self.cam_pose_seq.append(extrin)
                    self.seq_names.append(seq_path)
                    self.ids += max(0, len(images)-1)
                    self.seq_acc_ids.append(self.ids)

        if len(self.image_seq) == 0:
            raise Exception("The specified trajectory is not in the test set.")

    def __getitem__(self, index): 
        seq_idx = max(np.searchsorted(self.seq_acc_ids, index+1) - 1, 0)
        frame_idx = index - self.seq_acc_ids[seq_idx]

        this_idx = frame_idx
        next_idx = frame_idx + random.choice(self.keyframes)

        color0 = self.__load_rgb_tensor(self.image_seq[seq_idx][this_idx])
        color1 = self.__load_rgb_tensor(self.image_seq[seq_idx][next_idx])

        depth0 = self.__load_depth_tensor(self.depth_seq[seq_idx][this_idx])
        depth1 = self.__load_depth_tensor(self.depth_seq[seq_idx][next_idx])

        if self.transforms:
            color0, color1 = self.transforms([color0, color1])            

        # normalize the coordinate
        calib = np.asarray(self.calib[seq_idx], dtype=np.float32)
        calib[0] *= self.fx_s
        calib[1] *= self.fy_s
        calib[2] *= self.fx_s
        calib[3] *= self.fy_s

        cam_pose0 = self.cam_pose_seq[seq_idx][this_idx]
        cam_pose1 = self.cam_pose_seq[seq_idx][next_idx]
        transform = np.dot(np.linalg.inv(cam_pose1), cam_pose0).astype(np.float32)        

        name = '{:}_{:06d}to{:06d}'.format(self.seq_names[seq_idx], 
            this_idx, next_idx)

        return color0, color1, depth0, depth1, transform, calib, name

    def __len__(self): 
        return self.ids

    def __load_rgb_tensor(self, path):
        """ Load the rgb image
        """
        image = imread(path)[:, :, :3]
        image = image.astype(np.float32) / 255.0
        image = resize(image, None, fx=self.fx_s, fy=self.fy_s)
        return image

    def __load_depth_tensor(self, path):
        """ Load the depth:
            The depth images are scaled by a factor of 5000, i.e., a pixel 
            value of 5000 in the depth image corresponds to a distance of 
            1 meter from the camera, 10000 to 2 meter distance, etc. 
            A pixel value of 0 means missing value/no data.
        """
        depth = imread(path).astype(np.float32) / 5e3
        depth = resize(depth, None, fx=self.fx_s, fy=self.fy_s, interpolation=INTER_NEAREST)
        depth = np.clip(depth, a_min=0.5, a_max=5.0) # the accurate range of kinect depth
        return depth[np.newaxis, :]        

"""
Some utility files to work with the data
"""

def tq2mat(tq): 
    """ transform translation-quaternion (tq) to (4x4) matrix
    """
    tq = np.array(tq)
    T = np.eye(4)
    T[:3,:3] = quaternions.quat2mat(np.roll(tq[3:], 1))
    T[:3, 3] = tq[:3]
    return T

def write_sync_trajectory(local_dir, dataset, subject_name):
    """
    :param the root of the directory 
    :param the dataset category 'fr1', 'fr2' or 'fr3'
    """
    rgb_file  = osp.join(local_dir, dataset, subject_name, 'rgb.txt')
    depth_file= osp.join(local_dir, dataset, subject_name, 'depth.txt')
    pose_file = osp.join(local_dir, dataset, subject_name, 'groundtruth.txt')

    rgb_list = read_file_list(rgb_file)
    depth_list=read_file_list(depth_file)
    pose_list = read_file_list(pose_file)

    matches = associate_three(rgb_list, depth_list, pose_list, offset=0.0, max_difference=0.02)

    trajectory_info = []
    for (a,b,c) in matches:
        pose = [float(x) for x in pose_list[c]]
        rgb_file = osp.join(local_dir, dataset, subject_name, rgb_list[a][0])
        depth_file = osp.join(local_dir, dataset, subject_name, depth_list[b][0])
        trajectory_info.append([pose, rgb_file, depth_file])

    dataset_path = osp.join(local_dir, dataset, subject_name, 'sync_trajectory.pkl')

    with open(dataset_path, 'wb') as output:
        pickle.dump(trajectory_info, output)

    txt_path = osp.join(local_dir, dataset, subject_name, 'sync_trajectory.txt')
    pickle2txts(dataset_path, txt_path)

def pickle2txts(pickle_file, txt_file):
    '''
    write the pickle_file into a txt_file
    '''
    with open(pickle_file, 'rb') as pkl_file:
        traj = pickle.load(pkl_file)

    with open(txt_file, 'w') as f:
        for frame in traj:
            f.write(' '.join(['%f ' % x for x in frame[0]]))
            f.write(frame[1] + ' ')
            f.write(frame[2] + '\n')

"""
The following utility files are provided by TUM RGBD dataset benchmark

Refer: https://vision.in.tum.de/data/datasets/rgbd-dataset/tools
"""

def read_file_list(filename):
    """
    Reads a trajectory from a text file.

    File format:
    The file format is "stamp d1 d2 d3 ...", where stamp denotes the time stamp (to be matched)
    and "d1 d2 d3.." is arbitary data (e.g., a 3D position and 3D orientation) associated to this timestamp.

    Input:
    filename -- File name

    Output:
    dict -- dictionary of (stamp,data) tuples

    """
    file = open(filename)
    data = file.read()
    lines = data.replace(","," ").replace("\t"," ").split("\n")
    list = [[v.strip() for v in line.split(" ") if v.strip()!=""] for line in lines if len(line)>0 and line[0]!="#"]
    list = [(float(l[0]),l[1:]) for l in list if len(l)>1]
    return dict(list)

def associate(first_list, second_list,offset,max_difference):
    """
    Associate two dictionaries of (stamp,data). As the time stamps never match exactly, we aim
    to find the closest match for every input tuple.

    Input:
    first_list -- first dictionary of (stamp,data) tuples
    second_list -- second dictionary of (stamp,data) tuples
    offset -- time offset between both dictionaries (e.g., to model the delay between the sensors)
    max_difference -- search radius for candidate generation

    Output:
    matches -- list of matched tuples ((stamp1,data1),(stamp2,data2))

    """
    first_keys = first_list.keys()
    second_keys = second_list.keys()
    potential_matches = [(abs(a - (b + offset)), a, b)
                         for a in first_keys
                         for b in second_keys
                         if abs(a - (b + offset)) < max_difference]
    potential_matches.sort()
    matches = []
    for diff, a, b in potential_matches:
        if a in first_keys and b in second_keys:
            first_keys.remove(a)
            second_keys.remove(b)
            matches.append((a, b))

    matches.sort()
    return matches

def associate_three(first_list, second_list, third_list, offset, max_difference):
    """
    Associate two dictionaries of (stamp,data). As the time stamps never match exactly, we aim to find the closest match for every input tuple.

    Input:
    first_list -- first dictionary of (stamp,data) tuples (default to be rgb)
    second_list -- second dictionary of (stamp,data) tuples (default to be depth)
    third_list -- third dictionary of (stamp,data) tuples (default to be pose)
    offset -- time offset between both dictionaries (e.g., to model the delay between the sensors)
    max_difference -- search radius for candidate generation

    Output:
    matches -- list of matched tuples ((stamp1,data1),(stamp2,data2),(stamp3,data3))
    """

    first_keys = list(first_list)
    second_keys = list(second_list)
    third_keys = list(third_list)
    # find the potential matches in (rgb, depth)
    potential_matches_ab = [(abs(a - (b + offset)), a, b)
                            for a in first_keys
                            for b in second_keys
                            if abs(a - (b + offset)) < max_difference]
    potential_matches_ab.sort()
    matches_ab = []
    for diff, a, b in potential_matches_ab:
        if a in first_keys and b in second_keys:
            matches_ab.append((a, b))

    matches_ab.sort()

    # find the potential matches in (rgb, depth, pose)
    potential_matches = [(abs(a - (c + offset)), abs(b - (c + offset)), a,b,c)
                        for (a,b) in matches_ab
                        for c in third_keys
                        if abs(a - (c + offset)) < max_difference and
                        abs(b - (c + offset)) < max_difference]

    potential_matches.sort()
    matches_abc = []
    for diff_rgb, diff_depth, a, b, c in potential_matches:
        if a in first_keys and b in second_keys and c in third_keys:
            first_keys.remove(a)
            second_keys.remove(b)
            third_keys.remove(c)
            matches_abc.append((a,b,c))
    matches_abc.sort()
    return matches_abc

if __name__ == '__main__': 

    loader = TUM(category='test', keyframes=[1])

    import torchvision.utils as torch_utils
 
    torch_loader = data.DataLoader(loader, batch_size=16, 
        shuffle=False, num_workers=4)

    for batch in torch_loader: 
        color0, color1, depth0, depth1, transform, K, name = batch
        B, C, H, W = color0.shape

        bcolor0_img = torch_utils.make_grid(color0, nrow=4)

        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(bcolor0_img.numpy().transpose(1,2,0))
        plt.show()