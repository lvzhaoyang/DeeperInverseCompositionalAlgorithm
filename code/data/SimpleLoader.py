"""
This Simple loader partially refers to
https://github.com/NVlabs/learningrigidity/blob/master/SimpleLoader.py

@author: Zhaoyang Lv
@date: May, 2019
"""

import sys, os, random
import torch.utils.data as data
import os.path as osp

import numpy as np

from scipy.misc import imread

class SimpleLoader(data.Dataset):
    
    def __init__(self, color_dir, depth_dir, K):
        """
        :param the directory of color images
        :param the directory of depth images
        :param the intrinsic parameter [fx, fy, cx, cy]
        """

        print('This simple loader is designed for TUM. \n\
            The depth scale may be different in your depth format. ')

        color_files = sorted(os.listdir(color_dir))
        depth_files = sorted(os.listdir(depth_dir))

        # please ensure the two folders use the same number of synchronized files
        assert(len(color_files) == len(depth_files))

        self.color_pairs = []
        self.depth_pairs = []
        self.ids = len(color_files) - 1
        for idx in range(self.ids):        
            self.color_pairs.append([
                osp.join(color_dir, color_files[idx]), 
                osp.join(color_dir, color_files[idx+1])
                ] )
            self.depth_pairs.append([
                osp.join(depth_dir, depth_files[idx]), 
                osp.join(depth_dir, depth_files[idx+1])
                ] )

        self.K = K 

    def __getitem__(self, index):

        image0_path, image1_path = self.color_pairs[index]
        depth0_path, depth1_path = self.depth_pairs[index]

        image0 = self.__load_rgb_tensor(image0_path)
        image1 = self.__load_rgb_tensor(image1_path)

        depth0 = self.__load_depth_tensor(depth0_path)
        depth1 = self.__load_depth_tensor(depth1_path)

        calib = np.asarray(self.K, dtype=np.float32)

        return image0, image1, depth0, depth1, calib

    def __len__(self):
        return self.ids

    def __load_rgb_tensor(self, path):
        image = imread(path)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2,0,1))
        return image.astype(np.float32)

    def __load_depth_tensor(self, path):
        assert(path.endswith('.png'))
        depth = imread(path).astype(np.float32) / 5e3
        depth = np.clip(depth, a_min=0.5, a_max=5.0)

        return depth[np.newaxis, :]