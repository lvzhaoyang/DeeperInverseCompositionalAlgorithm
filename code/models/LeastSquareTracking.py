"""
The learned Inverse Compositional Tracking.
Support both ego-motion and object-motion tracking

@author: Zhaoyang Lv
@Date: March, 2019
"""

import torch
import torch.nn as nn
import numpy as np

from models.submodules import convLayer as conv
from models.submodules import color_normalize

from models.algorithms import TrustRegionBase as TrustRegion
from models.algorithms import ImagePyramids, DirectSolverNet, FeaturePyramid, DeepRobustEstimator

class LeastSquareTracking(nn.Module):

    # all enum types
    NONE                = -1
    RGB                 = 0

    CONV_RGBD           = 1
    CONV_RGBD2          = 2

    def __init__(self, encoder_name,
        max_iter_per_pyr,
        mEst_type,
        solver_type,
        tr_samples = 10,
        no_weight_sharing = False,
        timers = None):
        """
        :param the backbone network used for regression.
        :param the maximum number of iterations at each pyramid levels
        :param the type of weighting functions.
        :param the type of solver. 
        :param number of samples in trust-region solver
        :param True if we do not want to share weight at different pyramid levels
        :param (optional) time to benchmark time consumed at each step
        """
        super(LeastSquareTracking, self).__init__()

        self.construct_image_pyramids = ImagePyramids([0,1,2,3], pool='avg')
        self.construct_depth_pyramids = ImagePyramids([0,1,2,3], pool='max')

        self.timers = timers

        """ =============================================================== """
        """             Initialize the Deep Feature Extractor               """
        """ =============================================================== """

        if encoder_name == 'RGB':
            print('The network will use raw image as measurements.')
            self.encoder = None
            self.encoder_type = self.RGB
            context_dim = 1
        elif encoder_name == 'ConvRGBD':
            print('Use a network with RGB-D information \
            to extract the features')
            context_dim = 4
            self.encoder = FeaturePyramid(D=context_dim)
            self.encoder_type = self.CONV_RGBD
        elif encoder_name == 'ConvRGBD2':
            print('Use two stream network with two frame input')
            context_dim = 8
            self.encoder = FeaturePyramid(D=context_dim)
            self.encoder_type = self.CONV_RGBD2
        else:
            raise NotImplementedError()

        """ =============================================================== """
        """             Initialize the Robust Estimator                     """
        """ =============================================================== """

        if no_weight_sharing:
            self.mEst_func0 = DeepRobustEstimator(mEst_type)
            self.mEst_func1 = DeepRobustEstimator(mEst_type)
            self.mEst_func2 = DeepRobustEstimator(mEst_type)
            self.mEst_func3 = DeepRobustEstimator(mEst_type)
            mEst_funcs = [self.mEst_func0, self.mEst_func1, self.mEst_func2,
            self.mEst_func3]
        else:
            self.mEst_func = DeepRobustEstimator(mEst_type)
            mEst_funcs = [self.mEst_func, self.mEst_func, self.mEst_func,
            self.mEst_func]

        """ =============================================================== """
        """             Initialize the Trust-Region Damping                 """
        """ =============================================================== """

        if no_weight_sharing:
            # for residual volume, the input K is not assigned correctly
            self.solver_func0 = DirectSolverNet(solver_type, samples=tr_samples)
            self.solver_func1 = DirectSolverNet(solver_type, samples=tr_samples)
            self.solver_func2 = DirectSolverNet(solver_type, samples=tr_samples)
            self.solver_func3 = DirectSolverNet(solver_type, samples=tr_samples)
            solver_funcs = [self.solver_func0, self.solver_func1,
            self.solver_func2, self.solver_func3]
        else:
            self.solver_func = DirectSolverNet(solver_type, samples=tr_samples)
            solver_funcs = [self.solver_func, self.solver_func,
                self.solver_func, self.solver_func]

        """ =============================================================== """
        """             Initialize the Trust-Region Method                  """
        """ =============================================================== """

        self.tr_update0 = TrustRegion(max_iter_per_pyr,
            mEst_func   = mEst_funcs[0],
            solver_func = solver_funcs[0],
            timers      = timers)
        self.tr_update1 = TrustRegion(max_iter_per_pyr,
            mEst_func   = mEst_funcs[1],
            solver_func = solver_funcs[1],
            timers      = timers)
        self.tr_update2 = TrustRegion(max_iter_per_pyr,
            mEst_func   = mEst_funcs[2],
            solver_func = solver_funcs[2],
            timers      = timers)
        self.tr_update3 = TrustRegion(max_iter_per_pyr,
            mEst_func   = mEst_funcs[3],
            solver_func = solver_funcs[3],
            timers      = timers)

    def forward(self, img0, img1, depth0, depth1, K, init_only=False):
        """
        :input
        :param the reference image
        :param the target image
        :param the inverse depth of the reference image
        :param the inverse depth of the target image
        :param the pin-hole camera instrinsic (in vector) [fx, fy, cx, cy] 
        :param the initial pose [Rotation, translation]
        --------
        :return 
        :param estimated transform 
        """
        if self.timers: self.timers.tic('extract features')

        # pre-processing all the data, all the invalid inputs depth are set to 0
        invD0 = torch.clamp(1.0 / depth0, 0, 10)
        invD1 = torch.clamp(1.0 / depth1, 0, 10)
        invD0[invD0 == invD0.min()] = 0
        invD1[invD1 == invD1.min()] = 0
        invD0[invD0 == invD0.max()] = 0
        invD1[invD1 == invD1.max()] = 0
        
        I0 = color_normalize(img0)
        I1 = color_normalize(img1)

        x0 = self.__encode_features(I0, invD0, I1, invD1)
        x1 = self.__encode_features(I1, invD1, I0, invD0)
        d0 = self.construct_depth_pyramids(invD0)
        d1 = self.construct_depth_pyramids(invD1)

        if self.timers: self.timers.toc('extract features')

        poses_to_train = [[],[]] # '[rotation, translation]'
        B = invD0.shape[0]
        R0 = torch.eye(3,dtype=torch.float).expand(B,3,3).type_as(I0)
        t0 = torch.zeros(B,3,1,dtype=torch.float).type_as(I0)
        poseI = [R0, t0]

        # the prior of the mask
        prior_W = torch.ones(d0[3].shape).type_as(d0[3])

        if self.timers: self.timers.tic('trust-region update')
        # trust region update on level 3
        K3 = K >> 3
        output3 = self.tr_update3(poseI, x0[3], x1[3], d0[3], d1[3], K3, prior_W)
        pose3, mEst_W3 = output3[0], output3[1]
        poses_to_train[0].append(pose3[0])
        poses_to_train[1].append(pose3[1])
        # trust region update on level 2
        K2 = K >> 2
        output2 = self.tr_update2(pose3, x0[2], x1[2], d0[2], d1[2], K2, mEst_W3)
        pose2, mEst_W2 = output2[0], output2[1]
        poses_to_train[0].append(pose2[0])
        poses_to_train[1].append(pose2[1])
        # trust region update on level 1
        K1 = K >> 1
        output1 = self.tr_update1(pose2, x0[1], x1[1], d0[1], d1[1], K1, mEst_W2)
        pose1, mEst_W1 = output1[0], output1[1]
        poses_to_train[0].append(pose1[0])
        poses_to_train[1].append(pose1[1])
        # trust-region update on the raw scale
        output0 = self.tr_update0(pose1, x0[0], x1[0], d0[0], d1[0], K, mEst_W1)
        pose0 = output0[0]
        poses_to_train[0].append(pose0[0])
        poses_to_train[1].append(pose0[1])
        if self.timers: self.timers.toc('trust-region update')

        if self.training:
            pyr_R = torch.stack(tuple(poses_to_train[0]), dim=1)
            pyr_t = torch.stack(tuple(poses_to_train[1]), dim=1)                
            return pyr_R, pyr_t
        else:
            return pose0

    def __encode_features(self, img0, invD0, img1, invD1):
        """ get the encoded features
        """
        if self.encoder_type == self.RGB:
            # In the RGB case, we will only use the intensity image
            I = self.__color3to1(img0)
            x = self.construct_image_pyramids(I)
        elif self.encoder_type == self.CONV_RGBD:
            m = torch.cat((img0, invD0), dim=1)
            x = self.encoder.forward(m)
        elif self.encoder_type in [self.CONV_RGBD2]:
            m = torch.cat((img0, invD0, img1, invD1), dim=1)
            x = self.encoder.forward(m)
        else:
            raise NotImplementedError()

        x = [self.__Nto1(a) for a in x]

        return x

    def __Nto1(self, x):
        """ Take the average of multi-dimension feature into one dimensional,
            which boostrap the optimization speed
        """
        C = x.shape[1]
        return x.sum(dim=1, keepdim=True) / C

    def __color3to1(self, img):
        """ Return a gray-scale image
        """
        B, _, H, W = img.shape
        return (img[:,0] * 0.299 + img[:, 1] * 0.587 + img[:, 2] * 0.114).view(B,1,H,W)
