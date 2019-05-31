"""
Some training criterions

@author: Zhaoyang Lv
@date: March, 2019
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as func
import models.geometry as geo

def EPE3D_loss(input_flow, target_flow, invalid=None):
    """
    :param the estimated optical / scene flow
    :param the ground truth / target optical / scene flow
    :param the invalid mask, the mask has value 1 for all areas that are invalid
    """
    epe_map = torch.norm(target_flow-input_flow,p=2,dim=1)
    B = epe_map.shape[0]

    invalid_flow = (target_flow != target_flow) # check Nan same as torch.isnan

    mask = (invalid_flow[:,0,:,:] | invalid_flow[:,1,:,:] | invalid_flow[:,2,:,:]) 
    if invalid is not None:
        mask = mask | (invalid.view(mask.shape) > 0)

    epes = []
    for idx in range(B):
        epe_sample = epe_map[idx][~mask[idx].data]
        if len(epe_sample) == 0:
            epes.append(torch.zeros(()).type_as(input_flow))
        else:
            epes.append(epe_sample.mean()) 

    return torch.stack(epes)

def RPE(R, t):
    """ Calcualte the relative pose error 
    (a batch version of the RPE error defined in TUM RGBD SLAM TUM dataset)
    :param relative rotation
    :param relative translation
    """
    angle_error = geo.batch_mat2angle(R)
    trans_error = torch.norm(t, p=2, dim=1) 
    return angle_error, trans_error

def compute_RPE_loss(R_est, t_est, R_gt, t_gt):
    """
    :param estimated rotation matrix Bx3x3
    :param estimated translation vector Bx3
    :param ground truth rotation matrix Bx3x3
    :param ground truth translation vector Bx3
    """ 
    dR, dt = geo.batch_Rt_between(R_est, t_est, R_gt, t_gt)
    angle_error, trans_error = RPE(dR, dt)
    return angle_error, trans_error

def compute_RT_EPE_loss(R_est, t_est, R_gt, t_gt, depth0, K, invalid=None): 
    """ Compute the epe point error of rotation & translation
    :param estimated rotation matrix Bx3x3
    :param estimated translation vector Bx3
    :param ground truth rotation matrix Bx3x3
    :param ground truth translation vector Bx3
    :param reference depth image, 
    :param camera intrinsic 
    """
    
    loss = 0
    if R_est.dim() > 3: # training time [batch, num_poses, rot_row, rot_col]
        rH, rW = 60, 80 # we train the algorithm using a downsized input, (since the size of the input is not super important at training time)

        B,C,H,W = depth0.shape
        rdepth = func.interpolate(depth0, size=(rH, rW), mode='bilinear')
        rinvalid = func.interpolate(invalid.float(), size=(rH,rW), mode='bilinear')
        rK = K.clone()
        rK[:,0] *= float(rW) / W
        rK[:,1] *= float(rH) / H
        rK[:,2] *= float(rW) / W
        rK[:,3] *= float(rH) / H
        xyz = geo.batch_inverse_project(rdepth, rK)
        flow_gt = geo.batch_transform_xyz(xyz, R_gt, t_gt, get_Jacobian=False)

        for idx in range(R_est.shape[1]):
            flow_est= geo.batch_transform_xyz(xyz, R_est[:,idx], t_est[:,idx], get_Jacobian=False)
            loss += EPE3D_loss(flow_est, flow_gt.detach(), rinvalid) #* (1<<idx) scaling does not help that much
    else:
        xyz = geo.batch_inverse_project(depth0, K)
        flow_gt = geo.batch_transform_xyz(xyz, R_gt, t_gt, get_Jacobian=False)

        flow_est= geo.batch_transform_xyz(xyz, R_est, t_est, get_Jacobian=False)
        loss = EPE3D_loss(flow_est, flow_gt, invalid)

    return loss

