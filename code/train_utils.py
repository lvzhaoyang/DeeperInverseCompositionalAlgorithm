"""
The training utility functions

@author: Zhaoyang Lv
@date: March 2019
"""

import os, sys
from os.path import join
import torch
import torch.nn as nn

def check_cuda(items):
    if torch.cuda.is_available():
        return [x.cuda() for x in items]
    else:
        return items

def initialize_logger(opt, logfile_name):
    """ Initialize the logger for the network
    """
    from Logger import TensorBoardLogger
    log_dir = opt.dataset
    # if opt.resume_training:
    #     logfile_name = '_'.join([
    #         logfile_name,
    #         'resume'])

    log_dir = join('logs', log_dir, logfile_name)
    logger = TensorBoardLogger(log_dir, logfile_name)
    return logger

def create_optim(config, network):
    """ Create the optimizer
    """
    if config.opt=='sgd':
        optim = torch.optim.SGD(network.parameters(),
                    lr = config.lr,
                    momentum = 0.9,
                    weight_decay = 4e-4,
                    nesterov=False)
    elif config.opt=='adam' or config.opt=='sgdr':
        optim = torch.optim.Adam(network.parameters(),
                    lr = config.lr,
                    weight_decay = 4e-4
                    )
    elif config.opt=='rmsprop':
        optim = torch.optim.RMSprop(network.parameters(),
                    lr = config.lr,
                    weight_decay = 1e-4)
    else:
        raise NotImplementedError

    return optim

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_checkpoint_test(opt):
    if os.path.isfile(opt.checkpoint):
        print('=> loading checkpoint '+ opt.checkpoint)
        checkpoint = torch.load(opt.checkpoint)
    else:
        raise Exception('=> no checkpoint found at '+opt.checkpoint)
    return checkpoint

def load_checkpoint_train(opt):
    """ Loading the checking-point file if specified
    """
    checkpoint = None
    if opt.checkpoint:
        if os.path.isfile(opt.checkpoint):
            print('=> loading checkpoint '+ opt.checkpoint)

            checkpoint = torch.load(opt.checkpoint)
            print('=> loaded checkpoint '+ opt.checkpoint+' epoch %d'%checkpoint['epoch'])
            if opt.resume_training:
                opt.start_epoch = checkpoint['epoch']
                print('resume training on the checkpoint')
            else:
                print('start new training...')

            # This is to configure the module loaded from multi-gpu
            if opt.checkpoint_multigpu:
                from collections import OrderedDict
                state_dict_rename = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = k[7:] # remove `module.`
                    state_dict_rename[name] = v
                checkpoint['state_dict'] = state_dict_rename
        else:
            print('=> no checkpoint found at '+opt.checkpoint)
    return checkpoint

def set_learning_rate(optim, lr):
    """ manual set the learning rate for all specified parameters
    """
    for param_group in optim.param_groups:
        param_group['lr']=lr

def get_learning_rate(optim, name=None):
    """ retrieve the current learning rate
    """
    if name is None:
        # assume all the learning rate remains the same
        return optim.param_groups[0]['lr']

def adjust_learning_rate_manual(optim, epoch, lr, lr_decay_epochs, lr_decay_ratio):
    """ DIY the learning rate
    """
    for e in lr_decay_epochs:
        if epoch<e: break
        lr *= lr_decay_ratio
    set_learning_rate(optim, lr)
    return lr

def resize_input(img0, img1):
    """ Resize a pair of inputs
    """
    B, C, H, W = img0.shape
    resize_H = (H / 64) * 64
    resize_W = (W / 64) * 64
    if H != resize_H or W != resize_W:
        resize_img = nn.Upsample(size=(resize_H, resize_W),mode='bilinear')
        img0 = resize_img(img0)
        img1 = resize_img(img1)

    return img0, img1


"""
Deprecated Functions!!!
"""

def schedule_SGDR(optim, lr_min, lr_max, T_max, current_epoch, snapshots=None):
    """ Use SGD Restart method for learning rate scheduling
    """
    last_epoch = current_epoch % T_max - 1
    if last_epoch == 0:
        print('Restart SGD. Set learning rate to {:}'.format(lr_max))
        set_learning_rate(optim, lr_max)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max, lr_min, last_epoch)

    if snapshots is not None:
        # reload the snapshot using the best model on validation set
        print('Load snapshot')
        pass

    return scheduler

