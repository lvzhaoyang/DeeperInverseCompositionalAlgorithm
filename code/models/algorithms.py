"""
The algorithm backbone, primarily the three contributions proposed in our paper

@author: Zhaoyang Lv
@date: March, 2019
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as func

import models.geometry as geometry
from models.submodules import convLayer as conv
from models.submodules import fcLayer, initialize_weights

class TrustRegionBase(nn.Module):
    """ 
    This is the the base function of the trust-region based inverse compositional algorithm. 
    """
    def __init__(self,
        max_iter    = 3,
        mEst_func   = None,
        solver_func = None,
        timers      = None):
        """
        :param max_iter, maximum number of iterations
        :param mEst_func, the M-estimator function / network 
        :param solver_func, the trust-region function / network
        :param timers, if yes, counting time for each step
        """
        super(TrustRegionBase, self).__init__()

        self.max_iterations = max_iter
        self.mEstimator     = mEst_func
        self.directSolver   = solver_func
        self.timers         = timers

    def forward(self, pose, x0, x1, invD0, invD1, K, wPrior=None):
        """
        :param pose, the initial pose
            (extrinsic of the target frame w.r.t. the referenc frame)
        :param x0, the template features
        :param x1, the image features
        :param invD0, the template inverse depth
        :param invD1, the image inverse depth
        :param K, the intrinsic parameters, [fx, fy, cx, cy]
        :param wPrior (optional), provide an initial weight as input to the convolutional m-estimator
        """
        B, C, H, W = x0.shape
        px, py = geometry.generate_xy_grid(B,H,W,K)

        if self.timers: self.timers.tic('pre-compute Jacobians')
        J_F_p = self.precompute_Jacobian(invD0, x0, px, py, K)
        if self.timers: self.timers.toc('pre-compute Jacobians')

        if self.timers: self.timers.tic('compute warping residuals')
        residuals, occ = compute_warped_residual(pose, invD0, invD1, \
            x0, x1, px, py, K)
        if self.timers: self.timers.toc('compute warping residuals')

        if self.timers: self.timers.tic('robust estimator')
        weights = self.mEstimator(residuals, x0, x1, wPrior)
        wJ = weights.view(B,-1,1) * J_F_p
        if self.timers: self.timers.toc('robust estimator')

        if self.timers: self.timers.tic('pre-compute JtWJ')
        JtWJ = torch.bmm(torch.transpose(J_F_p, 1, 2) , wJ)
        if self.timers: self.timers.toc('pre-compute JtWJ')

        for idx in range(self.max_iterations):
            if self.timers: self.timers.tic('solve x=A^{-1}b')
            pose = self.directSolver(JtWJ,
                torch.transpose(J_F_p,1,2), weights, residuals,
                pose, invD0, invD1, x0, x1, K)
            if self.timers: self.timers.toc('solve x=A^{-1}b')
    
            if self.timers: self.timers.tic('compute warping residuals')
            residuals, occ = compute_warped_residual(pose, invD0, invD1, \
                x0, x1, px, py, K)
            if self.timers: self.timers.toc('compute warping residuals')

        return pose, weights

    def precompute_Jacobian(self, invD, x, px, py, K):
        """ Pre-compute the image Jacobian on the reference frame
        refer to equation (13) in the paper
        
        :param invD, template depth
        :param x, template feature
        :param px, normalized image coordinate in cols (x)
        :param py, normalized image coordinate in rows (y)
        :param K, the intrinsic parameters, [fx, fy, cx, cy]

        ------------
        :return precomputed image Jacobian on template
        """
        Jf_x, Jf_y = feature_gradient(x)
        Jx_p, Jy_p = compute_jacobian_warping(invD, K, px, py)
        J_F_p = compute_jacobian_dIdp(Jf_x, Jf_y, Jx_p, Jy_p)
        return J_F_p

class ImagePyramids(nn.Module):
    """ Construct the pyramids in the image / depth space
    """
    def __init__(self, scales, pool='avg'):
        super(ImagePyramids, self).__init__()
        if pool == 'avg':
            self.multiscales = [nn.AvgPool2d(1<<i, 1<<i) for i in scales]
        elif pool == 'max':
            self.multiscales = [nn.MaxPool2d(1<<i, 1<<i) for i in scales]
        else:
            raise NotImplementedError()

    def forward(self, x):
        x_out = [f(x) for f in self.multiscales]
        return x_out

class FeaturePyramid(nn.Module):
    """ 
    The proposed feature-encoder (A).
    It also supports to extract features using one-view only.
    """
    def __init__(self, D):
        super(FeaturePyramid, self).__init__()
        self.net0 = nn.Sequential(
            conv(True, D,  16, 3), 
            conv(True, 16, 32, 3, dilation=2),
            conv(True, 32, 32, 3, dilation=2))
        self.net1 = nn.Sequential(
            conv(True, 32, 32, 3),
            conv(True, 32, 64, 3, dilation=2),
            conv(True, 64, 64, 3, dilation=2))
        self.net2 = nn.Sequential(
            conv(True, 64, 64, 3),
            conv(True, 64, 96, 3, dilation=2),
            conv(True, 96, 96, 3, dilation=2))
        self.net3 = nn.Sequential(
            conv(True, 96, 96, 3),
            conv(True, 96, 128, 3, dilation=2),
            conv(True, 128,128, 3, dilation=2))

        initialize_weights(self.net0)
        initialize_weights(self.net1)
        initialize_weights(self.net2)
        initialize_weights(self.net3)
        self.downsample = torch.nn.AvgPool2d(kernel_size=2)

    def forward(self, x):
        x0 = self.net0(x)
        x0s= self.downsample(x0)
        x1 = self.net1(x0s)
        x1s= self.downsample(x1)
        x2 = self.net2(x1s)
        x2s= self.downsample(x2)
        x3 = self.net3(x2s)
        return x0, x1, x2, x3

class DeepRobustEstimator(nn.Module):
    """ The M-estimator 

    When use estimator_type = 'MultiScale2w', it is the proposed convolutional M-estimator
    """

    def __init__(self, estimator_type):
        super(DeepRobustEstimator, self).__init__()

        if estimator_type == 'MultiScale2w':
            self.D = 4
        elif estimator_type == 'None':
            self.mEst_func = self.__constant_weight
            self.D = -1
        else:
            raise NotImplementedError()

        if self.D > 0:
            self.net = nn.Sequential(
                conv(True, self.D, 16, 3, dilation=1),
                conv(True, 16, 32, 3, dilation=2),
                conv(True, 32, 64, 3, dilation=4),
                conv(True, 64, 1,  3, dilation=1),
                nn.Sigmoid() )
            initialize_weights(self.net)
        else:
            self.net = None

    def forward(self, residual, x0, x1, ws=None):
        """
        :param residual, the residual map
        :param x0, the feature map of the template
        :param x1, the feature map of the image
        :param ws, the initial weighted residual
        """
        if self.D == 1: # use residual only
            context = residual.abs()
            w = self.net(context)
        elif self.D == 4:
            B, C, H, W = residual.shape
            wl = func.interpolate(ws, (H,W), mode='bilinear', align_corners=True)
            context = torch.cat((residual.abs(), x0, x1, wl), dim=1)
            w = self.net(context)
        elif self.D < 0:
            w = self.mEst_func(residual)

        return w

    def __weight_Huber(self, x, alpha = 0.02):
        """ weight function of Huber loss:
        refer to P. 24 w(x) at
        https://members.loria.fr/moberger/Enseignement/Master2/Documents/ZhangIVC-97-01.pdf

        Note this current implementation is not differentiable.
        """
        abs_x = torch.abs(x)
        linear_mask = abs_x > alpha
        w = torch.ones(x.shape).type_as(x)

        if linear_mask.sum().item() > 0: 
            w[linear_mask] = alpha / abs_x[linear_mask]
        return w

    def __constant_weight(self, x):
        """ mimic the standard least-square when weighting function is constant
        """
        return torch.ones(x.shape).type_as(x)

class DirectSolverNet(nn.Module):

    # the enum types for direct solver
    SOLVER_NO_DAMPING       = 0
    SOLVER_RESIDUAL_VOLUME  = 1

    def __init__(self, solver_type, samples=10):
        super(DirectSolverNet, self).__init__()

        if solver_type == 'Direct-Nodamping':
            self.net = None
            self.type = self.SOLVER_NO_DAMPING
        elif solver_type == 'Direct-ResVol':
            # flattened JtJ and JtR (number of samples, currently fixed at 10)
            self.samples = samples
            self.net = deep_damping_regressor(D=6*6+6*samples)
            self.type = self.SOLVER_RESIDUAL_VOLUME
            initialize_weights(self.net)
        else: 
            raise NotImplementedError()

    def forward(self, JtJ, Jt, weights, R, pose0, invD0, invD1, x0, x1, K):
        """
        :param JtJ, the approximated Hessian JtJ
        :param Jt, the trasposed Jacobian
        :param weights, the weight matrix
        :param R, the residual
        :param pose0, the initial estimated pose
        :param invD0, the template inverse depth map
        :param invD1, the image inverse depth map
        :param x0, the template feature map
        :param x1, the image feature map
        :param K, the intrinsic parameters

        -----------
        :return updated pose
        """

        B = JtJ.shape[0]

        wR = (weights * R).view(B, -1, 1)
        JtR = torch.bmm(Jt, wR)

        if self.type == self.SOLVER_NO_DAMPING:
            # Add a small diagonal damping. Without it, the training becomes quite unstable
            # Do not see a clear difference by removing the damping in inference though
            diag_mask = torch.eye(6).view(1,6,6).type_as(JtJ)
            diagJtJ = diag_mask * JtJ
            traceJtJ = torch.sum(diagJtJ, (2,1))
            epsilon = (traceJtJ * 1e-6).view(B,1,1) * diag_mask
            Hessian = JtJ + epsilon
            pose_update = inverse_update_pose(Hessian, JtR, pose0)
        elif self.type == self.SOLVER_RESIDUAL_VOLUME:
            Hessian = self.__regularize_residual_volume(JtJ, Jt, JtR, weights,
                pose0, invD0, invD1, x0, x1, K, sample_range=self.samples)
            pose_update = inverse_update_pose(Hessian, JtR, pose0)
        else:
            raise NotImplementedError()

        return pose_update

    def __regularize_residual_volume(self, JtJ, Jt, JtR, weights, pose,
        invD0, invD1, x0, x1, K, sample_range):
        """ regularize the approximate with residual volume

        :param JtJ, the approximated Hessian JtJ
        :param Jt, the trasposed Jacobian
        :param JtR, the Right-hand size residual
        :param weights, the weight matrix
        :param pose, the initial estimated pose
        :param invD0, the template inverse depth map
        :param invD1, the image inverse depth map
        :param K, the intrinsic parameters
        :param x0, the template feature map
        :param x1, the image feature map
        :param sample_range, the numerb of samples

        ---------------
        :return the damped Hessian matrix
        """
        # the following current support only single scale
        JtR_volumes = []

        B, C, H, W = x0.shape
        px, py = geometry.generate_xy_grid(B, H, W, K)

        diag_mask = torch.eye(6).view(1,6,6).type_as(JtJ)
        diagJtJ = diag_mask * JtJ
        traceJtJ = torch.sum(diagJtJ, (2,1))
        epsilon = (traceJtJ * 1e-6).view(B,1,1) * diag_mask
        n = sample_range
        lambdas = torch.logspace(-5, 5, n).type_as(JtJ)

        for s in range(n):
            # the epsilon is to prevent the matrix to be too ill-conditioned
            D = lambdas[s] * diagJtJ + epsilon
            Hessian = JtJ + D
            pose_s = inverse_update_pose(Hessian, JtR, pose)

            res_s,_= compute_warped_residual(pose_s, invD0, invD1, x0, x1, px, py, K)
            JtR_s = torch.bmm(Jt, (weights * res_s).view(B,-1,1))
            JtR_volumes.append(JtR_s)

        JtR_flat = torch.cat(tuple(JtR_volumes), dim=2).view(B,-1)
        JtJ_flat = JtJ.view(B,-1)
        damp_est = self.net(torch.cat((JtR_flat, JtJ_flat), dim=1))
        R = diag_mask * damp_est.view(B,6,1) + epsilon # also lift-up

        return JtJ + R

def deep_damping_regressor(D):
    """ Output a damping vector at each dimension
    """
    net = nn.Sequential(
        fcLayer(in_planes=D,   out_planes=128, bias=True),
        fcLayer(in_planes=128, out_planes=256, bias=True),
        fcLayer(in_planes=256, out_planes=6, bias=True)
    ) # the last ReLU makes sure every predicted value is positive
    return net

def feature_gradient(img, normalize_gradient=True):
    """ Calculate the gradient on the feature space using Sobel operator
    :param the input image 
    -----------
    :return the gradient of the image in x, y direction
    """
    B, C, H, W = img.shape
    # to filter the image equally in each channel
    wx = torch.FloatTensor([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]]).view(1,1,3,3).type_as(img)
    wy = torch.FloatTensor([[-1,-2,-1],[ 0, 0, 0],[ 1, 2, 1]]).view(1,1,3,3).type_as(img)

    img_reshaped = img.view(-1, 1, H, W)
    img_pad = func.pad(img_reshaped, (1,1,1,1), mode='replicate')
    img_dx = func.conv2d(img_pad, wx, stride=1, padding=0)
    img_dy = func.conv2d(img_pad, wy, stride=1, padding=0)

    if normalize_gradient:
        mag = torch.sqrt((img_dx ** 2) + (img_dy ** 2)+ 1e-8)
        img_dx = img_dx / mag 
        img_dy = img_dy / mag

    return img_dx.view(B,C,H,W), img_dy.view(B,C,H,W)

def compute_jacobian_dIdp(Jf_x, Jf_y, Jx_p, Jy_p):
    """ chained gradient of image w.r.t. the pose
    :param the Jacobian of the feature map in x direction
    :param the Jacobian of the feature map in y direction
    :param the Jacobian of the x map to manifold p
    :param the Jacobian of the y map to manifold p
    ------------
    :return the image jacobian in x, y, direction, Bx2x6 each
    """
    B, C, H, W = Jf_x.shape

    # precompute J_F_p, JtWJ
    Jf_p = Jf_x.view(B,C,-1,1) * Jx_p.view(B,1,-1,6) + \
        Jf_y.view(B,C,-1,1) * Jy_p.view(B,1,-1,6)
    
    return Jf_p.view(B,-1,6)

def compute_jacobian_warping(p_invdepth, K, px, py):
    """ Compute the Jacobian matrix of the warped (x,y) w.r.t. the inverse depth
    (linearized at origin)
    :param p_invdepth the input inverse depth
    :param the intrinsic calibration
    :param the pixel x map
    :param the pixel y map
     ------------
    :return the warping jacobian in x, y direction
    """
    B, C, H, W = p_invdepth.size()
    assert(C == 1)

    x = px.view(B, -1, 1)
    y = py.view(B, -1, 1)
    invd = p_invdepth.view(B, -1, 1)

    xy = x * y
    O = torch.zeros((B, H*W, 1)).type_as(p_invdepth)

    # This is cascaded Jacobian functions of the warping function
    # Refer to the supplementary materials for math documentation
    dx_dp = torch.cat((-xy,     1+x**2, -y, invd, O, -invd*x), dim=2)
    dy_dp = torch.cat((-1-y**2, xy,     x, O, invd, -invd*y), dim=2)

    fx, fy, cx, cy = torch.split(K, 1, dim=1)
    return dx_dp*fx.view(B,1,1), dy_dp*fy.view(B,1,1)

def compute_warped_residual(pose, invD0, invD1, x0, x1, px, py, K, obj_mask=None):
    """ Compute the residual error of warped target image w.r.t. the reference feature map.
    refer to equation (12) in the paper

    :param the forward warping pose from the reference camera to the target frame.
        Note that warping from the target frame to the reference frame is the inverse of this operation.
    :param the reference inverse depth
    :param the target inverse depth
    :param the reference feature image
    :param the target feature image
    :param the pixel x map
    :param the pixel y map
    :param the intrinsic calibration
    -----------
    :return the residual (of reference image), and occlusion information
    """
    u_warped, v_warped, inv_z_warped = geometry.batch_warp_inverse_depth(
        px, py, invD0, pose, K)
    x1_1to0 = geometry.warp_features(x1, u_warped, v_warped)
    occ = geometry.check_occ(inv_z_warped, invD1, u_warped, v_warped)

    residuals = x1_1to0 - x0 # equation (12)

    B, C, H, W = x0.shape
    if obj_mask is not None:
        # determine whether the object is in-view
        occ = occ & (obj_mask.view(B,1,H,W) < 1)
    residuals[occ.expand(B,C,H,W)] = 1e-3

    return residuals, occ

def inverse_update_pose(H, Rhs, pose):
    """ Ues left-multiplication for the pose update 
    in the inverse compositional form
    refer to equation (10) in the paper 

    :param the (approximated) Hessian matrix
    :param Right-hand side vector
    :param the initial pose (forward transform inverse of xi)
    ---------
    :return the forward updated pose (inverse of xi)
    """
    inv_H = invH(H)
    xi = torch.bmm(inv_H, Rhs)
    # simplifed inverse compositional for SE3
    d_R = geometry.batch_twist2Mat(-xi[:, :3].view(-1,3))
    d_t = -torch.bmm(d_R, xi[:, 3:])

    R, t = pose
    pose = geometry.batch_Rt_compose(R, t, d_R, d_t) 
    return pose

def invH(H):
    """ Generate (H+damp)^{-1}, with predicted damping values
    :param approximate Hessian matrix JtWJ
    -----------
    :return the inverse of Hessian
    """
    # GPU is much slower for matrix inverse when the size is small (compare to CPU)
    # works (50x faster) than inversing the dense matrix in GPU
    if H.is_cuda:
        # invH = bpinv((H).cpu()).cuda()
        # invH = torch.inverse(H)
        invH = torch.inverse(H.cpu()).cuda()
    else:
        invH = torch.inverse(H)
    return invH
