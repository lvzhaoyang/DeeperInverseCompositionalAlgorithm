""" The dataloaders for training and evaluation

@author: Zhaoyang Lv
@date: March 2019
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torchvision.transforms as transforms
import numpy as np

def load_data(dataset_name, keyframes = None, load_type = 'train',
    select_trajectory = '', load_numpy = False):
    """ Use two frame camera pose data loader
    """
    if select_trajectory == '':
        select_trajectory = None

    if not load_numpy:
        if load_type == 'train': 
            data_transform = image_transforms(['color_augment', 'numpy2torch'])
        else:
            data_transform = image_transforms(['numpy2torch'])
    else:
        data_transform = image_transforms([])

    if dataset_name == 'TUM_RGBD':
        from data.TUM_RGBD import TUM
        np_loader = TUM('data', load_type, keyframes, data_transform, select_trajectory)
    elif dataset_name == 'MovingObjects3D': 
        from data.MovingObj3D import MovingObjects3D
        np_loader = MovingObjects3D('data', load_type, keyframes, data_transform)
    # elif dataset_name == 'BundleFusion':
    #     from data.BundleFusion import BundleFusion
    #     np_loader = BundleFusion(load_type, keyframes, data_transform)
    # elif dataset_name == 'Refresh':
    #     from data.REFRESH import REFRESH
    #     np_loader = REFRESH(load_type, keyframes)
    else:
        raise NotImplementedError()

    return np_loader

def image_transforms(options):

    transform_list = []

    if 'color_augment' in options: 
        augment_parameters = [0.9, 1.1, 0.9, 1.1, 0.9, 1.1]
        transform_list.append(AugmentImages(augment_parameters))

    if 'numpy2torch' in options:
        transform_list.append(ToTensor())

    # if 'color_normalize' in options: # we do it on the fly
    #     transform_list.append(ColorNormalize())

    return transforms.Compose(transform_list)

class ColorNormalize(object):

    def __init__(self):
        rgb_mean = (0.4914, 0.4822, 0.4465)
        rgb_std = (0.2023, 0.1994, 0.2010)
        self.transform = transforms.Normalize(mean=rgb_mean, std=rgb_std)

    def __call__(self, sample):
        return [self.transform(x) for x in sample]

class ToTensor(object):
    def __init__(self):
        self.transform = transforms.ToTensor()

    def __call__(self, sample):
        return [self.transform(x) for x in sample] 

class AugmentImages(object):
    def __init__(self, augment_parameters):
        self.gamma_low  = augment_parameters[0]         # 0.9
        self.gamma_high = augment_parameters[1]         # 1.1
        self.brightness_low  = augment_parameters[2]    # 0.9
        self.brightness_high = augment_parameters[3]    # 1,1
        self.color_low  = augment_parameters[4]         # 0.9
        self.color_high = augment_parameters[5]         # 1.1

        self.thresh = 0.5

    def __call__(self, sample):
        p = np.random.uniform(0, 1, 1)
        if p > self.thresh:
            random_gamma = np.random.uniform(self.gamma_low, self.gamma_high)
            random_brightness = np.random.uniform(self.brightness_low, self.brightness_high)
            random_colors = np.random.uniform(self.color_low, self.color_high, 3)
            for x in sample:
                x = x ** random_gamma             # randomly shift gamma
                x = x * random_brightness         # randomly shift brightness
                for i in range(3):                # randomly shift color
                    x[:, :, i] *= random_colors[i]
                    x[:, :, i] *= random_colors[i]
                x = np.clip(x, a_min=0, a_max=1)  # saturate
            return sample
        else:        
            return sample
