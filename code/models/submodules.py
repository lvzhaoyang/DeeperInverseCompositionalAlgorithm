"""
Submodules to build up CNN

@author: Zhaoyang Lv
@date: March, 2019
"""

from __future__ import print_function

import torch.nn as nn
import torch
import numpy as np

from torch.nn import init
from torchvision import transforms

def color_normalize(color):
    rgb_mean = torch.Tensor([0.4914, 0.4822, 0.4465]).type_as(color)
    rgb_std = torch.Tensor([0.2023, 0.1994, 0.2010]).type_as(color)
    return (color - rgb_mean.view(1,3,1,1)) / rgb_std.view(1,3,1,1)

def convLayer(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, dilation=1, bias=False):
    """ A wrapper of convolution-batchnorm-ReLU module
    """
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2 + dilation-1, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_planes),
            #nn.LeakyReLU(0.1,inplace=True) # deprecated 
            nn.ELU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2 + dilation-1, dilation=dilation, bias=True),
            #nn.LeakyReLU(0.1,inplace=True) # deprecated
            nn.ELU(inplace=True)
        )

def fcLayer(in_planes, out_planes, bias=True):
    return nn.Sequential(
        nn.Linear(in_planes, out_planes, bias),
        nn.ReLU(inplace=True)
    )

def initialize_weights(modules, method='xavier'):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                m.bias.data.zero_()
            if method == 'xavier':
                init.xavier_uniform_(m.weight)
            elif method == 'kaiming':
                init.kaiming_uniform_(m.weight)

        if isinstance(m, nn.ConvTranspose2d):
            if m.bias is not None:
                m.bias.data.zero_()
            if method == 'xavier':
                init.xavier_uniform_(m.weight)
            elif method == 'kaiming':
                init.kaiming_uniform_(m.weight)
                
class ListModule(nn.Module):
    """ The implementation of a list of modules from
    https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)
