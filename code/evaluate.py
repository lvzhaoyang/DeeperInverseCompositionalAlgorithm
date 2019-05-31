""" 
Evaluation scripts

@author: Zhaoyang Lv
@date: March 2019
"""

import os, sys, argparse, pickle
import os.path as osp
import numpy as np
import pandas as pd

import torch
import torch.utils.data as data
import torchvision.utils as torch_utils
import torch.nn as nn

import models.LeastSquareTracking as ICtracking
import models.criterions as criterions
import train_utils
import config

from data.dataloader import load_data
from timers import Timers
from tqdm import tqdm

def check_directory(filename):
    target_dir = os.path.dirname(filename)
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)

def eval_trajectories(dataset):
    return {
        'TUM_RGBD': ['fr1/rgbd_dataset_freiburg1_360', 
            'fr1/rgbd_dataset_freiburg1_desk', 
            'fr2/rgbd_dataset_freiburg2_desk', 
            'fr2/rgbd_dataset_freiburg2_pioneer_360']
    }[dataset]

def create_eval_loaders(options, eval_type, keyframes, 
    total_batch_size = 8, 
    trajectory  = ''):
    """ create the evaluation loader at different keyframes set-up
    """
    eval_loaders = {}

    if trajectory == '': 
        trajectories = eval_trajectories(options.dataset)
    else: 
        trajectories = [trajectory]

    for trajectory in trajectories:
        for kf in keyframes:
            np_loader = load_data(options.dataset, [kf], eval_type, trajectory)
            eval_loaders['{:}_keyframe_{:}'.format(trajectory, kf)] = data.DataLoader(np_loader, 
                batch_size = int(total_batch_size),
                shuffle = False, num_workers = options.cpu_workers)
    
    return eval_loaders

def evaluate_trust_region(dataloader, net, objectives, eval_name='',
        known_mask = False, timers = None):
    """ evaluate the trust-region method given the two-frame pose estimation
    :param the pytorch dataloader
    :param the network
    :param the evaluation objective names, e.g. RPE, EPE3D
    :param True if ground mask if known
    :param (optional) timing each step
    """

    progress = tqdm(dataloader, ncols=100,
        desc = 'evaluate deeper inverse compositional algorithm {:}'.format(eval_name),
        total= len(dataloader))

    net.eval()

    total_frames = len(dataloader.dataset)

    outputs = {
        'R_est': np.zeros((total_frames, 3, 3)),
        't_est': np.zeros((total_frames, 3)),
        'names': []
    }
    flow_loss, rpe_loss = None, None
    if 'EPE3D' in objectives: 
        flow_loss = criterions.compute_RT_EPE_loss
        outputs['epes'] = np.zeros(total_frames)
    if 'RPE' in objectives:
        rpe_loss = criterions.compute_RPE_loss
        outputs['angular_error'] = np.zeros(total_frames)  
        outputs['translation_error'] = np.zeros(total_frames) 

    count_base = 0

    if timers: timers.tic('one iteration')

    count = 1
    for idx, batch in enumerate(progress):

        if timers: timers.tic('forward step')

        names = batch[-1]

        if known_mask: # for dataset that with mask or need mask
            color0, color1, depth0, depth1, Rt, K, obj_mask0, obj_mask1 = \
                train_utils.check_cuda(batch[:8])
        else:
            color0, color1, depth0, depth1, Rt, K = \
                train_utils.check_cuda(batch[:6])
            obj_mask0, obj_mask1 = None, None

        B, _, H, W = depth0.shape
            
        with torch.no_grad():
            output = net.forward(color0, color1, depth0, depth1, K)
            R, t = output

        if timers: timers.toc('forward step')

        outputs['R_est'][count_base:count_base+B] = R.cpu().numpy()
        outputs['t_est'][count_base:count_base+B] = t.cpu().numpy()

        if timers: timers.tic('evaluate')
        R_gt, t_gt = Rt[:,:3,:3], Rt[:,:3,3]
        if rpe_loss: # evaluate the relative pose error             
            angle_error, trans_error = rpe_loss(R, t, R_gt, t_gt)
            outputs['angular_error'][count_base:count_base+B] = angle_error.cpu().numpy()
            outputs['translation_error'][count_base:count_base+B] = trans_error.cpu().numpy()

        if flow_loss:# evaluate the end-point-error loss 3D
            invalid_mask = (depth0 == depth0.min()) + (depth0 == depth0.max())
            if obj_mask0 is not None: invalid_mask = ~obj_mask0 + invalid_mask

            epes3d = flow_loss(R, t, R_gt, t_gt, depth0, K, invalid_mask)            
            outputs['epes'][count_base:count_base+B] = epes3d.cpu().numpy()

        outputs['names'] += names

        count_base += B

        if timers: timers.toc('evaluate')
        if timers: timers.toc('one iteration')
        if timers: timers.tic('one iteration')

    # if timers: timers.print()

    return outputs

def test_TrustRegion(options):

    if options.time:
        timers = Timers()
    else:
        timers = None

    print('Evaluate test performance with the (deep) direct method.')

    total_batch_size = options.batch_per_gpu *  torch.cuda.device_count()

    keyframes = [int(x) for x in options.keyframes.split(',')]
    if options.dataset in ['BundleFusion', 'TUM_RGBD']:
        obj_has_mask = False
    else:
        obj_has_mask = True

    eval_loaders = create_eval_loaders(options, options.eval_set,
        keyframes, total_batch_size, options.trajectory)

    if options.checkpoint == '':
        print('No checkpoint loaded. Use the non-learning method')
        net = ICtracking.LeastSquareTracking(
            encoder_name    = 'RGB',
            max_iter_per_pyr= options.max_iter_per_pyr,
            mEst_type       = 'None',
            solver_type     = 'Direct-Nodamping')
        if torch.cuda.is_available(): net.cuda()
        net.eval()
    else:
        train_utils.load_checkpoint_test(options)

        net = ICtracking.LeastSquareTracking(
            encoder_name    = options.encoder_name,
            max_iter_per_pyr= options.max_iter_per_pyr,
            mEst_type       = options.mestimator,
            solver_type     = options.solver,
            no_weight_sharing = options.no_weight_sharing)

        if torch.cuda.is_available(): net.cuda()
        net.eval()

        # check whether it is a single checkpoint or a directory
        net.load_state_dict(torch.load(options.checkpoint)['state_dict'])

    eval_objectives = ['EPE3D', 'RPE']

    output_prefix = '_'.join([
        options.network,
        options.encoder_name,
        options.mestimator,
        options.solver,
        'iter', str(options.max_iter_per_pyr)
    ])

    # evaluate results per trajectory per key-frame
    outputs = {}
    for k, loader in eval_loaders.items():

        traj_name, kf = k.split('_keyframe_')

        output_name = '{:}_{:}'.format(output_prefix, k)
        info = evaluate_trust_region(loader, net,
            eval_objectives,
            eval_name = 'tmp/'+output_name,
            known_mask=obj_has_mask)

        # collect results 
        outputs[k] = pd.Series([info['epes'].mean(), 
            info['angular_error'].mean(), 
            info['translation_error'].mean(), 
            info['epes'].shape[0], int(kf), traj_name], 
            index=['3D EPE', 'axis error', 'trans error', 'total frames', 'keyframe', 'trajectory'])

        print(outputs[k])

        checkpoint_name = options.checkpoint.replace('.pth.tar', '')
        if checkpoint_name == '':
            checkpoint_name = 'nolearning'
        output_dir = osp.join(options.eval_set+'_results', checkpoint_name, k)
        output_pkl = output_dir + '.pkl'
        
        check_directory(output_pkl)

        with open(output_pkl, 'wb') as output: # dump per-frame results info
            info = info
            pickle.dump(info, output)

    """ =============================================================== """
    """             Generate the final evaluation results                   """
    """ =============================================================== """

    outputs_pd = pd.DataFrame(outputs).T
    outputs_pd['3D EPE'] *= 100 # convert to cm
    outputs_pd['axis error'] *= (180/np.pi) # convert to degree
    outputs_pd['trans error'] *= 100 # convert to cm

    print(outputs_pd)

    stats_dict = {}
    for kf in keyframes:        
        kf_outputs = outputs_pd[outputs_pd['keyframe']==kf]

        stats_dict['mean values of trajectories keyframe {:}'.format(kf)] = pd.Series(
            [kf_outputs['3D EPE'].mean(), 
             kf_outputs['axis error'].mean(),
             kf_outputs['trans error'].mean(), kf], 
            index=['3D EPE', 'axis error', 'trans error', 'keyframe'])

        total_frames = kf_outputs['total frames'].sum()
        stats_dict['mean values of frames keyframe {:}'.format(kf)] = pd.Series(
            [(kf_outputs['3D EPE'] * kf_outputs['total frames']).sum() / total_frames, 
             (kf_outputs['axis error'] * kf_outputs['total frames']).sum() / total_frames, 
             (kf_outputs['trans error']* kf_outputs['total frames']).sum() / total_frames, kf],
            index=['3D EPE', 'axis error', 'trans error', 'keyframe'])

    stats_pd = pd.DataFrame(stats_dict).T
    print(stats_pd)

    final_pd = outputs_pd.append(stats_pd, sort=False)
    final_pd.to_csv('{:}.csv'.format(output_dir))

    return outputs_pd

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Evaluate the network")
    config.add_basics_config(parser)
    config.add_test_basics_config(parser)
    config.add_tracking_config(parser)

    options = parser.parse_args()

    print('---------------------------------------')

    outputs = test_TrustRegion(options)

