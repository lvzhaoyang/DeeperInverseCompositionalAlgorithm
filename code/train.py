"""
The training script for deep trust region method

@author: Zhaoyang Lv 
@date: March 2019
"""

import os, sys, argparse, time

import models.LeastSquareTracking as ICtracking
import models.criterions as criterions
import models.geometry as geometry
import train_utils
import config
from data.dataloader import load_data

import torch
import torch.nn as nn
import torch.utils.data as data

from timers import Timers
from tqdm import tqdm

import evaluate as eval_utils

def create_train_eval_loaders(options, eval_type, keyframes, 
    total_batch_size = 8, 
    trajectory  = ''):
    """ create the evaluation loader at different keyframes set-up
    """
    eval_loaders = {}

    for kf in keyframes:
        np_loader = load_data(options.dataset, [kf], eval_type, trajectory)
        eval_loaders['{:}_keyframe_{:}'.format(trajectory, kf)] = data.DataLoader(np_loader, 
            batch_size = int(total_batch_size),
            shuffle = False, num_workers = options.cpu_workers)
    
    return eval_loaders

def train_one_epoch(options, dataloader, net, optim, epoch, logger, objectives,
    known_mask=False, timers=None):

    net.train()

    epoch_len = len(dataloader)

    flow_loss, rpe_loss = None, None
    if 'EPE3D' in objectives: 
        flow_loss = criterions.compute_RT_EPE_loss
    if 'RPE' in objectives:
        rpe_loss = criterions.compute_RPE_loss

    if timers is None: timers_iter = Timers()

    if timers: timers.tic('one iteration')
    else: timers_iter.tic('one iteration')

    for batch_idx, batch in enumerate(dataloader):

        display_dict = {}

        optim.zero_grad()

        if timers: timers.tic('forward step')

        if known_mask: # for dataset that with mask or need mask
            color0, color1, depth0, depth1, Rt, K, obj_mask0, obj_mask1 = \
                train_utils.check_cuda(batch[:8])
        else:
            color0, color1, depth0, depth1, Rt, K = \
                train_utils.check_cuda(batch[:6])
            obj_mask0, obj_mask1 = None, None

        # Bypass lazy way to bypass invalid pixels. 
        invalid_mask = (depth0 == depth0.min()) + (depth0 == depth0.max())
        if obj_mask0 is not None:
            invalid_mask = 1.0 - obj_mask0 + invalid_mask

        Rs, ts = net.forward(color0, color1, depth0, depth1, K)[:2]

        if timers: timers.toc('forward step')
        if timers: timers.tic('calculate loss')

        R_gt, t_gt = Rt[:,:3,:3], Rt[:,:3,3]
 
        assert(flow_loss) # the only loss used for training

        epes3d = flow_loss(Rs, ts, R_gt, t_gt, depth0, K, invalid_mask).mean() * 1e2
        display_dict['train_epes3d'] = epes3d.item()

        loss = epes3d
        display_dict['train_loss'] = loss.item()

        if timers: timers.toc('calculate loss')
        if timers: timers.tic('backward')

        loss.backward()

        if timers: timers.toc('backward')

        optim.step()

        lr = train_utils.get_learning_rate(optim)
        display_dict['lr'] = lr

        if timers:
            timers.toc('one iteration')
            batch_time = timers.get_avg('one iteration')
            timers.tic('one iteration')
        else:
            timers_iter.toc('one iteration')
            batch_time = timers_iter.get_avg('one iteration')
            timers_iter.tic('one iteration')

        logger.write_to_tensorboard(display_dict, epoch*epoch_len + batch_idx)
        logger.write_to_terminal(display_dict, epoch, batch_idx, epoch_len, batch_time, is_train=True)

def train(options):

    if options.time:
        timers = Timers()
    else:
        timers = None

    total_batch_size = options.batch_per_gpu *  torch.cuda.device_count()

    checkpoint = train_utils.load_checkpoint_train(options)

    keyframes = [int(x) for x in options.keyframes.split(',')]
    train_loader = load_data(options.dataset, keyframes, load_type = 'train')
    train_loader = data.DataLoader(train_loader,
        batch_size = total_batch_size,
        shuffle = True, num_workers = options.cpu_workers)
    if options.dataset in ['BundleFusion', 'TUM_RGBD']:
        obj_has_mask = False
    else:
        obj_has_mask = True

    eval_loaders = create_train_eval_loaders(options, 'validation', keyframes, total_batch_size)

    logfile_name = '_'.join([
        options.prefix, # the current test version
        options.network,
        options.encoder_name,
        options.mestimator,
        options.solver,
        'lr', str(options.lr),
        'batch', str(total_batch_size),
        'kf', options.keyframes])

    print("Initialize and train the Deep Trust Region Network")
    net = ICtracking.LeastSquareTracking(
        encoder_name    = options.encoder_name,
        max_iter_per_pyr= options.max_iter_per_pyr,
        mEst_type       = options.mestimator,
        solver_type     = options.solver,
        tr_samples      = options.tr_samples,
        no_weight_sharing = options.no_weight_sharing,
        timers          = timers)

    if options.no_weight_sharing:
        logfile_name += '_no_weight_sharing'
    logger = train_utils.initialize_logger(options, logfile_name)

    if options.checkpoint:
        net.load_state_dict(checkpoint['state_dict'])

    if torch.cuda.is_available(): net.cuda()

    net.train()

    if torch.cuda.device_count() > 1:
        print("Use", torch.cuda.device_count(), "GPUs for training!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        net = nn.DataParallel(net)

    train_objective = ['EPE3D'] # Note: we don't use RPE for training
    eval_objectives = ['EPE3D', 'RPE']

    num_params = train_utils.count_parameters(net)

    if num_params < 1:
        print('There is no learnable parameters in this baseline.')
        print('No training. Only one iteration of evaluation')
        no_training = True
    else:
        print('There is a total of {:} learnabled parameters'.format(num_params))
        no_training = False
        optim = train_utils.create_optim(options, net)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim,
            milestones=options.lr_decay_epochs,
            gamma=options.lr_decay_ratio)

    freq = options.save_checkpoint_freq
    for epoch in range(options.start_epoch, options.epochs):

        if epoch % freq == 0:
            checkpoint_name = 'checkpoint_epoch{:d}.pth.tar'.format(epoch)
            print('save {:}'.format(checkpoint_name))
            state_info = {'epoch': epoch, 'num_param': num_params}
            logger.save_checkpoint(net, state_info, filename=checkpoint_name)

        if options.no_val is False:
            for k, loader in eval_loaders.items():

                eval_name = '{:}_{:}'.format(options.dataset, k)

                eval_info = eval_utils.evaluate_trust_region(
                    loader, net, eval_objectives, 
                    known_mask  = obj_has_mask, 
                    eval_name   = eval_name,
                    timers      = timers)

                display_dict = {"{:}_epe3d".format(eval_name): eval_info['epes'].mean(), 
                    "{:}_rpe_angular".format(eval_name): eval_info['angular_error'].mean(), 
                    "{:}_rpe_translation".format(eval_name): eval_info['translation_error'].mean()}

                logger.write_to_tensorboard(display_dict, epoch)

        if no_training: break

        train_one_epoch(options, train_loader, net, optim, epoch, logger,
            train_objective, known_mask=obj_has_mask, timers=timers)

        scheduler.step()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training the network')

    config.add_basics_config(parser)
    config.add_train_basics_config(parser)
    config.add_train_optim_config(parser)
    config.add_train_log_config(parser)
    config.add_train_loss_config(parser)
    config.add_tracking_config(parser)

    options = parser.parse_args()

    options.start_epoch = 0

    print('---------------------------------------')
    print_options = vars(options)
    for key in print_options.keys():
        print(key+': '+str(print_options[key]))
    print('---------------------------------------')

    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    print('Start training...')
    train(options)
