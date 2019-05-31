""" 
Add the configurations by modules

@author: Zhaoyang Lv
@date: March 2019
"""

def add_tracking_config(parser):
    parser.add_argument('--network',
        default='DeepIC', type=str,
        choices=('DeepIC', 'GaussNewton'),
        help='Choose a network to run. \n \
        The DeepIC is the proposed Deeper Inverse Compositional method. \n\
        The GuassNewton is the baseline for Inverse Compositional method which does not include \
        any learnable parameters\n')
    parser.add_argument('--mestimator',
        default='MultiScale2w', type=str,
        choices=('None', 'MultiScale2w'),
        help='Choose a weighting function for the Trust Region method.\n\
            The MultiScale2w is the proposed (B) convolutional M-estimator. \n')
    parser.add_argument('--solver',
        default='Direct-ResVol', type=str,
        choices=('Direct-Nodamping', 'Direct-ResVol'),
        help='Choose the solver function for the Trust Region method. \n\
            Direct-Nodamping is the Gauss-Newton algorithm, which does not use damping. \n\
            Direct-ResVol is the proposed (C) Trust-Region Network. \n\
            (default: Direct-ResVol) ')
    parser.add_argument('--encoder_name',
        default='ConvRGBD2',
        choices=('ConvRGBD2', 'RGB'),
        help='The encoder architectures. \
            ConvRGBD2 takes the two-view features as input. \n\
            RGB is using the raw RGB images as input (converted to intensity afterwards).\n\
            (default: ConvRGBD2)')
    parser.add_argument('--max_iter_per_pyr',
        default=3, type=int,
        help='The maximum number of iterations at each pyramids.\n')
    parser.add_argument('--no_weight_sharing',
        action='store_true',
        help='If this flag is on, we disable sharing the weights across different backbone network when extracing \
         features. In default, we share the weights for all network in each pyramid level.\n')
    parser.add_argument('--tr_samples', default=10, type=int,
        help='Set the number of trust-region samples. (default: 10)\n')

def add_basics_config(parser):
    """ the basic setting
    (supposed to be shared through train and inference)
    """
    parser.add_argument('--cpu_workers', type=int, default=12,
        help="Number of cpu threads for data loader.\n")
    parser.add_argument('--dataset', type=str,
        choices=('TUM_RGBD', 'MovingObjects3D'),
        help='Choose a dataset to train/val/evaluate.\n')
    parser.add_argument('--time', dest='time', action='store_true',
        help='Count the execution time of each step.\n' )

def add_test_basics_config(parser):
    parser.add_argument('--batch_per_gpu', default=8, type=int,
        help='Specify the batch size during test. The default is 8.\n')
    parser.add_argument('--checkpoint', default='', type=str,
        help='Choose a checkpoint model to test.\n')
    parser.add_argument('--keyframes',
        default='1,2,4,8', type=str,
        help='Choose the number of keyframes to train the algorithm.\n')
    parser.add_argument('--verbose', action='store_true',
        help='Print/save all the intermediate representations')
    parser.add_argument('--eval_set', default='test',
        choices=('test', 'validation'))
    parser.add_argument('--trajectory', type=str, 
        default = '',
        help = 'Specify a trajectory to run.\n')

def add_train_basics_config(parser):
    """ add the basics about the training """
    parser.add_argument('--checkpoint', default='', type=str,
        help='Choose a pretrained checkpoint model to start with. \n')
    parser.add_argument('--batch_per_gpu', default=64, type=int,
        help='Specify the batch size during training.\n')
    parser.add_argument('--epochs',
        default=30, type=int,
        help='The total number of total epochs to run. Default is 30.\n' )
    parser.add_argument('--resume_training',
        dest='resume_training', action='store_true',
        help='Resume the training using the loaded checkpoint. If not, restart the training. \n\
            You will need to use the --checkpoint config to load the pretrained checkpoint' )
    parser.add_argument('--pretrained_model', default='', type=str,
        help='Initialize the model weights with pretrained model.\n')
    parser.add_argument('--no_val',
        default=False,
        action='store_true',
        help='Use no validatation set for training.\n')
    parser.add_argument('--keyframes',
        default='1,2,4,8', type=str,
        help='Choose the number of keyframes to train the algorithm')
    parser.add_argument('--verbose', action='store_true',
        help='Print/save all the intermediate representations.\n')

def add_train_log_config(parser):
    """ checkpoint and log options """
    parser.add_argument('--checkpoint_folder', default='', type=str,
        help='The folder name (postfix) to save the checkpoint.')
    parser.add_argument('--snapshot', default=1, type=int,
        help='Number of interations to save a snapshot')
    parser.add_argument('--save_checkpoint_freq',
        default=1, type=int,
        help='save the checkpoint for every N epochs')
    parser.add_argument('--prefix', default='', type=str,
        help='the prefix string added to the log files')
    parser.add_argument('-p', '--print_freq',
        default=10, type=int,
        help='print frequency (default: 10)')

def add_train_optim_config(parser):
    """ add training optimization options """
    parser.add_argument('--opt',
        type=str, default='adam', choices=('sgd','adam'),
        help='choice of optimizer (default: adam) \n')
    parser.add_argument('--lr',
        default=0.0005, type=float,
        help='initial learning rate. \n')
    parser.add_argument('--lr_decay_ratio',
        default=0.5, type=float,
        help='lr decay ratio (default:0.5)')
    parser.add_argument('--lr_decay_epochs',
        default=[5, 10, 20], type=int, nargs='+',
        help='lr decay epochs')
    parser.add_argument('--lr_min', default=1e-6, type=float,
        help='minimum learning rate')
    parser.add_argument('--lr_restart', default=10, type=int,
        help='restart learning after N epochs')

def add_train_loss_config(parser):
    """ add training configuration for the loss function """
    parser.add_argument('--regression_loss_type',
        default='SmoothL1', type=str, choices=('L1', 'SmoothL1'),
        help='Loss function for flow regression (default: SmoothL1 loss)')