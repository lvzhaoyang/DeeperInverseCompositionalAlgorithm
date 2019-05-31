""" Logger, wrapped on Tensorboard 1.0.0a6
Tensorboard is not backward compatible since then. 

@author: Zhaoyang Lv
@date: March 2019
"""

import sys, os, shutil
import os.path as osp
import tensorboard
import torch
from collections import OrderedDict

class Logger(object):
    """
    example usage:

        stdout = Logger('log.txt')
        sys.stdout = stdout

        ... your code here ...

        stdout.delink()
    """
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def delink(self):
        self.log.close()
        #self.log = open('foo', "w")
#        self.write = self.writeTerminalOnly

    def writeTerminalOnly(self, message):
        self.terminal.write(message)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

class TensorBoardLogger(object):
    def __init__(self, logging_dir, logfile_name, print_freq = 10):

        self.log_dir = logging_dir
        self.print_freq = print_freq

        if not os.path.isdir(logging_dir):
            os.makedirs(logging_dir)

        self.summary_writer = tensorboard.FileWriter(logdir=logging_dir)

        # standard logger to print to terminal
        logfile = osp.join(logging_dir,'log.txt')
        stdout = Logger(logfile)
        sys.stdout = stdout

    def write_to_tensorboard(self, display_dict, iteration):
        """ Write the saved states (display_dict) to tensorboard
        """
        for k, v in display_dict.items():
            self.summary_writer.add_summary(tensorboard.summary.scalar(k, v), iteration)

    def write_to_terminal(self, display_dict, epoch, batch_iter, epoch_len, batch_time, is_train = True):
        """ Write the save states (display_dict) and training information to terminal for display
        """

        if batch_iter % self.print_freq != 0:
            return

        if is_train:
            prefix = 'Train'
        else:
            prefix = 'Test'

        state = prefix + ':\tEpoch %d, Batch %d/%d, BatchTime %.4f'%(epoch+1, batch_iter, epoch_len, batch_time)

        loss = ''
        for k, v in display_dict.items():
            loss += k + ' ' + '%.8f ' % v

        print(state + loss)

    def save_checkpoint(self, network, state_info = None,
        filename='checkpoint.pth.tar'):
        """save checkpoint to disk"""
        state_dict = network.state_dict().copy()

        if torch.cuda.device_count() > 1:
            state_dict_rename = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                state_dict_rename[name] = v
            state_dict = state_dict_rename

        if state_info is None:
            state = {'state_dict': state_dict}
        else:
            state = state_info
            state['state_dict'] = state_dict

        checkpoint_path = osp.join(self.log_dir,filename)
        torch.save(state, checkpoint_path)
        return checkpoint_path
