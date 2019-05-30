""" Add the configurations by modules
"""

def add_tracking_config(parser):
    parser.add_argument('--network',
        default='DeepIC', type=str,
        choices=('DeepIC', 'GaussNewton'),
        help='Choose a network to run. \n \
        The DeepIC is the proposed Deeper Inverse Compositional method. \n\
        The GuassNewton is the baseline for Inverse Compositional method which does not include \
        any learnable parameters')
    parser.add_argument('--mestimator',
        default='MultiScale2w', type=str,
        choices=('None', 'MultiScale2w'),
        help='Choose a weighting function for the Trust Region method')
    parser.add_argument('--solver',
        default='Direct-ResVol', type=str,
        choices=('Direct-Nodamping', 'Direct-ResVol'),
        help='Choose the solver function for the Trust Region method.')
    parser.add_argument('--encoder_name',
        default='ConvRGBD2',
        choices=('ConvRGBD2', 'ConvRGBD', 'RGB'),
        help='The encoder architectures. \
            ConvRGBD2 takes the two-view features as input. \n \
            ConvRGB only takes the reference view feature as input.')
    parser.add_argument('--max_iter_per_pyr',
        default=3, type=int,
        help='The maximum number of iterations at each pyramids')
    parser.add_argument('--no_weight_sharing',
        action='store_true',
        help='Share the weights across different backbone network when extracing \
         features (in both vanilla conv net and least-square tracking net).')
    parser.add_argument('--tr_samples', default=10, type=int,
        help='set the trust-region samples')

def add_basics_config(parser):
    """ the basic setting
    (supposed to be shared through train and inference)
    """
    parser.add_argument('--cpu_workers', type=int, default=12,
        help="Number of cpu threads for data loader")
    parser.add_argument('--dataset', type=str,
        choices=('BundleFusion', 'Refresh', 'FlyObjs', 'TUM_RGBD'),
        help='Choose a dataset to train/val/evaluate.')
    parser.add_argument('--time', dest='time', action='store_true',
        help='count the execution time of each step' )

def add_test_basics_config(parser):
    parser.add_argument('--batch_per_gpu', default=8, type=int,
        help='specify the batchsize during test')
    parser.add_argument('--checkpoint', default='', type=str,
        help='choose a checkpoint model to test')
    parser.add_argument('--keyframes',
        default='1,2,4,8', type=str,
        help='choose the number of keyframes to train the algorithm')
    parser.add_argument('--verbose', action='store_true',
        help='print/save all the intermediate representations')
    parser.add_argument('--eval_set', default='test',
        choices=('test', 'validation'))
    parser.add_argument('--trajectory', type=str, 
        default = '',
        help = 'specify a trajectory to run')

def add_train_basics_config(parser):
    """ add the basics about the training """
    parser.add_argument('--checkpoint', default='', type=str,
        help='choose a checkpoint model to start with')
    parser.add_argument('--batch_per_gpu', default=24, type=int,
        help='specify the batchsize during training')
    parser.add_argument('--epochs',
        default=1000, type=int,
        help='The total number of total epochs to run (defaul: 1000)' )
    parser.add_argument('--resume_training',
        dest='resume_training', action='store_true',
        help='resume training on the loaded checkpoint' )
    parser.add_argument('--pretrained_model', default='', type=str,
        help='initialize the model weights with pretrained model')
    parser.add_argument('--no_val',
        default=False,
        action='store_true',
        help='Use no validatation set for training')
    parser.add_argument('--keyframes',
        default='1,2,4,8', type=str,
        help='choose the number of keyframes to train the algorithm')
    parser.add_argument('--verbose', action='store_true',
        help='print/save all the intermediate representations')

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
        type=str, default='adam', choices=('sgd','adam','rmsprop', 'sgdr'),
        help='choice of optimizer (default: adam)')
    parser.add_argument('--lr',
        default=0.0005, type=float,
        help='initial learning rate (useful only for sgd)')
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