"""
Data loader for MovingObjs 3D dataset

@author: Zhaoyang Lv
@date: May 2019
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys, os, random
import pickle
import functools

import numpy as np
import torch.utils.data as data
import os.path as osp

from scipy.misc import imread
from tqdm import tqdm

from cv2 import resize, INTER_NEAREST

class MovingObjects3D(data.Dataset):

    # every sequence has 200 frames.
    categories = {
        'train': {'aeroplane':  [0,190],
                'bicycle':      [0,190],
                'bus':          [0,190],
                'car':          [0,190]},

        'validation': {'aeroplane': [190,200],
                'bicycle':          [190,200],
                'bus':              [190,200],
                'car':              [190,200]},

        'test': {'boat':           [0,200],
                'motorbike':        [0,200]}
    }

    def __init__(self, root, load_type='train', keyframes = [1], data_transform=None):
        super(MovingObjects3D, self).__init__()

        self.base_folder = osp.join(root, 'data_objs3D')

        data_all = self.categories[load_type]

        # split it into train and test set (the first 20 are for test)
        self.image_seq = []
        self.depth_seq = []
        self.invalid_seq = []
        self.object_mask_seq = []
        self.cam_pose_seq = []
        self.obj_pose_seq = []
        self.obj_vis_idx = []
        self.calib = []
        self.obj_names = []

        self.transforms = data_transform

        if load_type in ['validation', 'test']:
            # should not mix different keyframes in test
            assert(len(keyframes) == 1)
            self.keyframes = [1]
            self.sample_freq = keyframes[0]
        else:
            self.keyframes = keyframes
            self.sample_freq = 1

        self.ids = 0
        self.images_size = [240, 320]
        # get the accumulated image sequences on the fly
        self.seq_acc_ids = [0]
        for data_obj, frame_interval in data_all.items():
            start_idx, end_idx = frame_interval
            print('Load {:} data from frame {:d} to {:d}'.format(data_obj, start_idx, end_idx))
            for seq_idx in range(start_idx, end_idx, 1):
                seq_str = "{:06d}".format(seq_idx)

                info_pkl= osp.join(self.base_folder,
                    data_obj, seq_str, 'info.pkl')

                color_seq, depth_seq, invalid_seq, mask_seq, camera_poses_seq, object_poses_seq,\
                    obj_visible_frames, calib_seq = extract_info_pickle(info_pkl)

                obj_visible_frames = obj_visible_frames[::self.sample_freq]

                self.image_seq.append([osp.join(self.base_folder, x) for x in color_seq]) 
                self.depth_seq.append([osp.join(self.base_folder, x) for x in depth_seq])
                # self.invalid_seq.append(invalid_seq)
                self.object_mask_seq.append([osp.join(self.base_folder, x) for x in mask_seq])
                self.cam_pose_seq.append(camera_poses_seq)
                self.obj_pose_seq.append(object_poses_seq)
                self.calib.append(calib_seq)
                self.obj_vis_idx.append(obj_visible_frames)

                self.obj_names.append('{:}_{:03d}'.format(data_obj, seq_idx))

                total_valid_frames = max(0, len(obj_visible_frames) - max(self.keyframes))

                self.ids += total_valid_frames
                self.seq_acc_ids.append(self.ids)

        # downscale the input image to half
        self.fx_s = 0.5
        self.fy_s = 0.5

        print('There are a total of {:} valid frames'.format(self.ids))

    def __len__(self):
        return self.ids

    def __getitem__(self, index):
        # the index we want from search sorted is shifted for one
        seq_idx = max(np.searchsorted(self.seq_acc_ids, index+1) - 1, 0)
        frame_idx = index - self.seq_acc_ids[seq_idx]

        this_idx= self.obj_vis_idx[seq_idx][frame_idx]
        next_idx= self.obj_vis_idx[seq_idx][frame_idx + random.choice(self.keyframes)]

        color0 = self.__load_rgb_tensor(self.image_seq[seq_idx][this_idx])
        color1 = self.__load_rgb_tensor(self.image_seq[seq_idx][next_idx])

        if self.transforms:
            color0, color1 = self.transforms([color0, color1])

        depth0 = self.__load_depth_tensor(self.depth_seq[seq_idx][this_idx])
        depth1 = self.__load_depth_tensor(self.depth_seq[seq_idx][next_idx])

        cam_pose0 = self.cam_pose_seq[seq_idx][this_idx]
        cam_pose1 = self.cam_pose_seq[seq_idx][next_idx]
        obj_pose0 = self.obj_pose_seq[seq_idx][this_idx]
        obj_pose1 = self.obj_pose_seq[seq_idx][next_idx]

        # the relative allocentric transform of objects
        transform = functools.reduce(np.dot,
        [np.linalg.inv(cam_pose1), obj_pose1, np.linalg.inv(obj_pose0), cam_pose0]).astype(np.float32)

        # the validity of the object is up the object mask
        obj_index = 1 # object index is in default to be 1
        obj_mask0 = self.__load_binary_mask_tensor(self.object_mask_seq[seq_idx][this_idx], obj_index)
        obj_mask1 = self.__load_binary_mask_tensor(self.object_mask_seq[seq_idx][next_idx], obj_index)

        calib = np.asarray(self.calib[seq_idx], dtype=np.float32)
        calib[0] *= self.fx_s
        calib[1] *= self.fy_s
        calib[2] *= self.fx_s
        calib[3] *= self.fy_s

        obj_name = self.obj_names[seq_idx]
        pair_name = '{:}/{:06d}to{:06d}'.format(obj_name, this_idx, next_idx)

        return color0, color1, depth0, depth1, transform, calib, obj_mask0, obj_mask1, pair_name

    def __load_rgb_tensor(self, path):
        """ Load the rgb image
        """
        image = imread(path)[:, :, :3]
        image = image.astype(np.float32) / 255.0
        image = resize(image, None, fx=self.fx_s, fy=self.fy_s)
        return image

    def __load_depth_tensor(self, path):
        """ Load the depth
        """
        depth = imread(path).astype(np.float32) / 1e3
        depth = resize(depth, None, fx=self.fx_s, fy=self.fy_s, interpolation=INTER_NEAREST)
        depth = np.clip(depth, 1e-1, 1e2) # the valid region of the depth
        return depth[np.newaxis, :]

    def __load_binary_mask_tensor(self, path, seg_index):
        """ Load a binary segmentation mask (numbers)
            If the object matches the specified index, return true;
            Otherwise, return false
        """
        obj_mask = imread(path)
        mask = (obj_mask == seg_index)
        mask = resize(mask.astype(np.float), None, fx=self.fx_s, fy=self.fy_s, interpolation=INTER_NEAREST)
        return mask.astype(np.uint8)[np.newaxis, :]

def extract_info_pickle(info_pkl): 

    with open(info_pkl, 'rb') as p:
        info = pickle.load(p)

        color_seq = [x.split('final/')[1] for x in info['color']]
        depth_seq = [x.split('final/')[1] for x in info['depth']]
        invalid_seq = [x.split('final/')[1] for x in info['invalid'] ]
        mask_seq = [x.split('final/')[1] for x in info['object_mask']]

        # in this rendering setting, there is only one object
        camera_poses_seq = info['pose']
        object_poses_seq = info['object_poses']['Model_1']
        object_visible_frames = info['object_visible_frames']['Model_1']

        calib_seq = info['calib']

    return color_seq, depth_seq, invalid_seq, mask_seq, \
        camera_poses_seq, object_poses_seq, object_visible_frames, calib_seq


if __name__ == '__main__':

    loader = MovingObjects3D('', load_type='train', keyframes=[1])

    import torchvision.utils as torch_utils
    torch_loader = data.DataLoader(loader, batch_size=16, shuffle=False, num_workers=4)

    for batch in torch_loader:
        color0, color1, depth0, depth1, transform, K, mask0, mask1, names = batch
        B,C,H,W=color0.shape

        bcolor0_img = torch_utils.make_grid(color0, nrow=4)
        bcolor1_img = torch_utils.make_grid(color1, nrow=4)
        # bdepth0_img = torch_utils.make_grid(depth0, nrow=4)
        # bdepth1_img = torch_utils.make_grid(depth1, nrow=4)
        bmask0_img = torch_utils.make_grid(mask0.view(B,1,H,W)*255, nrow=4)
        bmask1_img = torch_utils.make_grid(mask1.view(B,1,H,W)*255, nrow=4)

        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(bcolor0_img.numpy().transpose((1,2,0)))
        plt.figure()
        plt.imshow(bcolor1_img.numpy().transpose((1,2,0)))
        # plt.figure()
        # plt.imshow(bdepth0_img.numpy().transpose((1,2,0)))
        # plt.figure()
        # plt.imshow(bdepth1_img.numpy().transpose((1,2,0)))
        plt.figure()
        plt.imshow(bmask0_img.numpy().transpose((1,2,0)))
        plt.figure()
        plt.imshow(bmask1_img.numpy().transpose((1,2,0)))
        plt.show()

