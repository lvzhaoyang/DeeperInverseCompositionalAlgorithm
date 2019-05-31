"""
An extremely simple example to show how to run the algorithm

@author: Zhaoyang Lv
@date: May 2019
"""

import argparse 

import torch
import torch.nn as nn
import torch.nn.functional as func

import models.LeastSquareTracking as ICtracking

from tqdm import tqdm
from torch.utils.data import DataLoader
from train_utils import check_cuda
from data.SimpleLoader import SimpleLoader

def resize(img0, img1, depth0, depth1, K_in, resizeH, resizeW): 
    H, W = img0.shape[-2:]

    I0 = func.interpolate(img0, (resizeH,resizeW), mode='bilinear', align_corners=True)
    I1 = func.interpolate(img1, (resizeH,resizeW), mode='bilinear', align_corners=True)
    D0 = func.interpolate(depth0, (resizeH,resizeW), mode='nearest')
    D1 = func.interpolate(depth1, (resizeH,resizeW), mode='nearest')

    sx = resizeH / H
    sy = resizeW / W

    K = K_in.clone()
    K[:,0] *= sx
    K[:,1] *= sy
    K[:,2] *= sx
    K[:,3] *= sy

    return I0, I1, D0, D1, K

def run_inference(dataloader, net):

    progress = tqdm(dataloader, ncols=100, 
        desc = 'Run the deeper inverse compositional algorithm', 
        total= len(dataloader))

    net.eval()

    for idx, batch, in enumerate(progress): 

        color0, color1, depth0, depth1, K = check_cuda(batch)

        # downsize the input to 120*160, it is the size of data when the algorthm is trained
        C0, C1, D0, D1, K = resize(color0, color1, depth0, depth1, K, resizeH = 120, resizeW=160)

        with torch.no_grad():
            R, t = net.forward(C0, C1, D0, D1, K)

        print('Rotation: ')
        print(R)
        print('translation: ')
        print(t)

if __name__ == '__main__': 

    parser = argparse.ArgumentParser(description='Run the network inference example.')

    parser.add_argument('--checkpoint', default='trained_models/TUM_RGBD_ABC_final.pth.tar', 
        type=str, help='the path to the pre-trained checkpoint.')

    parser.add_argument('--color_dir', default='data/data_examples/TUM/color',
        help='the directory of color images')
    parser.add_argument('--depth_dir', default='data/data_examples/TUM/depth', 
        help='the directory of depth images')

    parser.add_argument('--intrinsic', default='525.0,525.0,319.5,239.5', 
        help='Simple pin-hole camera intrinsics, input in the format (fx, fy, cx, cy)')

    config = parser.parse_args()
    
    K = [float(x) for x in config.intrinsic.split(',')]

    simple_loader = SimpleLoader(config.color_dir, config.depth_dir, K)
    simple_loader = DataLoader(simple_loader, batch_size=1, shuffle=False)

    net = ICtracking.LeastSquareTracking(
        encoder_name    = 'ConvRGBD2',
        max_iter_per_pyr= 3,
        mEst_type       = 'MultiScale2w',
        solver_type     = 'Direct-ResVol')

    if torch.cuda.is_available(): 
        net.cuda()
    
    net.load_state_dict(torch.load(config.checkpoint)['state_dict'])

    run_inference(simple_loader, net)