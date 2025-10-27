import argparse


def option():
    # Training settings
    parser = argparse.ArgumentParser(description='HAIMNet')
    parser.add_argument('--batchSize', type=int, default=8, help='training batch size')
    parser.add_argument('--cropSize', type=int, default=256, help='image crop size (patch size)')
    parser.add_argument('--nEpochs', type=int, default=1000, help='number of epochs to train for')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='epoch to start from; >0 to resume from a checkpoint')
    parser.add_argument('--snapshots', type=int, default=10, help='interval of epochs to save checkpoints')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--gpu_mode', type=bool, default=True, help='use GPU if available')
    parser.add_argument('--shuffle', type=bool, default=True, help='shuffle training data')
    parser.add_argument('--threads', type=int, default=16, help='number of data loading threads')

    # Scheduler settings
    parser.add_argument('--cos_restart_cyclic', type=bool, default=False, help='use cyclic cosine annealing restart')
    parser.add_argument('--cos_restart', type=bool, default=True, help='use single cosine annealing restart')
    parser.add_argument('--warmup_epochs', type=int, default=50, help='number of warmup epochs')
    parser.add_argument('--start_warmup', type=bool, default=True, help='enable warmup before scheduler')

    # Data paths for training
    parser.add_argument('--data_train_lol_blur', type=str, default='./datasets/LOL_blur/train',
                        help='LOL_blur training path')
    parser.add_argument('--data_train_lol_v1', type=str, default='./datasets/LOLv1/Train', help='LOLv1 training path')
    parser.add_argument('--data_train_lolv2_real', type=str, default='./datasets/LOLv2/Real_captured/Train',
                        help='LOLv2 real training path')
    parser.add_argument('--data_train_lolv2_syn', type=str, default='./datasets/LOLv2/Synthetic/Train',
                        help='LOLv2 synthetic training path')
    parser.add_argument('--data_train_SID', type=str, default='./datasets/Sony_total_dark/train',
                        help='SID training path')
    parser.add_argument('--data_train_SICE', type=str, default='./datasets/SICE/Dataset/train',
                        help='SICE training path')

    # Data paths for validation input
    parser.add_argument('--data_val_lol_blur', type=str, default='./datasets/LOL_blur/eval/low_blur',
                        help='LOL_blur validation input path')
    parser.add_argument('--data_val_lol_v1', type=str, default='./datasets/LOLv1/Test/input',
                        help='LOLv1 validation input path')
    parser.add_argument('--data_val_lolv2_real', type=str, default='./datasets/LOLv2/Real_captured/Test/Low',
                        help='LOLv2 real validation input path')
    parser.add_argument('--data_val_lolv2_syn', type=str, default='./datasets/LOLv2/Synthetic/Test/Low',
                        help='LOLv2 synthetic validation input path')
    parser.add_argument('--data_val_SID', type=str, default='./datasets/Sony_total_dark/eval/short',
                        help='SID validation input path')
    parser.add_argument('--data_val_SICE_mix', type=str, default='./datasets/SICE/Dataset/eval/test',
                        help='SICE mix validation input path')
    parser.add_argument('--data_val_SICE_grad', type=str, default='./datasets/SICE/Dataset/eval/test',
                        help='SICE grad validation input path')

    # Data paths for validation ground truth
    parser.add_argument('--data_valgt_lol_blur', type=str, default='./datasets/LOL_blur/eval/high_sharp_scaled/',
                        help='LOL_blur validation GT path')
    parser.add_argument('--data_valgt_lol_v1', type=str, default='./datasets/LOLv1/Test/target/',
                        help='LOLv1 validation GT path')
    parser.add_argument('--data_valgt_lolv2_real', type=str, default='./datasets/LOLv2/Real_captured/Test/Normal/',
                        help='LOLv2 real validation GT path')
    parser.add_argument('--data_valgt_lolv2_syn', type=str, default='./datasets/LOLv2/Synthetic/Test/Normal/',
                        help='LOLv2 synthetic validation GT path')
    parser.add_argument('--data_valgt_SID', type=str, default='./datasets/Sony_total_dark/eval/long/',
                        help='SID validation GT path')
    parser.add_argument('--data_valgt_SICE_mix', type=str, default='./datasets/SICE/Dataset/eval/target/',
                        help='SICE mix validation GT path')
    parser.add_argument('--data_valgt_SICE_grad', type=str, default='./datasets/SICE/Dataset/eval/target/',
                        help='SICE grad validation GT path')

    # Output directories
    parser.add_argument('--checkpoint_dir', type=str, default='./weights/train',
                        help='directory to save checkpoints and TensorBoard logs')
    parser.add_argument('--val_folder', type=str, default='./results/', help='directory to save validation outputs')
    parser.add_argument('--metrics_dir', type=str, default='./results/metrics',
                        help='directory to save markdown metrics')

    # Loss weights
    parser.add_argument('--HVI_weight', type=float, default=1.0)
    parser.add_argument('--L1_weight', type=float, default=0.0)
    parser.add_argument('--D_weight',  type=float, default=0.0)
    parser.add_argument('--E_weight',  type=float, default=0.0)
    parser.add_argument('--P_weight',  type=float, default=0.01)

    parser.add_argument('--cgaf_lr', type=float, default=2e-4,
                        help='learning rate for CGAF modules (e.g., cgaf_i3/cgaf_h3)')

    # Gamma augmentation
    parser.add_argument('--gamma', type=bool, default=False, help='enable random gamma augmentation')
    parser.add_argument('--start_gamma', type=int, default=60, help='start value for gamma range')
    parser.add_argument('--end_gamma', type=int, default=120, help='end value for gamma range')

    # Grad options
    parser.add_argument('--grad_detect', type=bool, default=False, help='enable anomaly detection for gradients')
    parser.add_argument('--grad_clip', type=bool, default=True, help='enable gradient clipping')

    # Dataset selection (only one should be True)
    parser.add_argument('--lol_v1', type=bool, default=False, help='train on LOL v1')
    parser.add_argument('--lolv2_real', type=bool, default=True, help='train on LOL v2 real')
    parser.add_argument('--lolv2_syn', type=bool, default=False, help='train on LOL v2 synthetic')
    parser.add_argument('--lol_blur', type=bool, default=False, help='train on LOL blur')
    parser.add_argument('--SID', type=bool, default=False, help='train on SID')
    parser.add_argument('--SICE_mix', type=bool, default=False, help='train on SICE mix')
    parser.add_argument('--SICE_grad', type=bool, default=False, help='train on SICE grad')

    opt =  parser.parse_args()

    if opt.lol_v1:
        opt.checkpoint_dir = './weights/train_lolv1'
        opt.val_folder = './results/lolv1/'
        opt.metrics_dir = './results/metrics_lolv1/'
    elif opt.lolv2_real:
        opt.checkpoint_dir = './weights/train_lolv2real'
        opt.val_folder = './results/lolv2real/'
        opt.metrics_dir = './results/metrics_lolv2real/'
    elif opt.lolv2_syn:
        opt.checkpoint_dir = './weights/train_lolv2syn'
        opt.val_folder = './results/lolv2syn/'
        opt.metrics_dir = './results/metrics_lolv2syn/'
        # …你所有其它数据集的分支…
    elif opt.SID:
        opt.checkpoint_dir = './weights/train_SID'
        opt.val_folder = './results/SID/'
        opt.metrics_dir = './results/metrics_SID/'
    elif opt.lol_blur:
        opt.checkpoint_dir = './weights/train_lol_blur'
        opt.val_folder = './results/lol_blur/'
        opt.metrics_dir = './results/metrics_lol_blur/'
    elif opt.SICE_mix:
        opt.checkpoint_dir = './weights/train_SICE_mix'
        opt.val_folder = './results/SICE_mix/'
        opt.metrics_dir = './results/metrics_SICE_mix/'
    elif opt.SICE_grad:
        opt.checkpoint_dir = './weights/train_SICE_grad'
        opt.val_folder = './results/SICE_grad/'
        opt.metrics_dir = './results/metrics_SICE_grad/'
    else:
        raise ValueError("请指定一个数据集开关，如 --lol_v1")

    return opt