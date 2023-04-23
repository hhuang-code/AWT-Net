import argparse


# settings
import pdb

parser = argparse.ArgumentParser(description='3D shape part segmentation')
parser.add_argument('--exp_name', type=str, default='AWT-Net')
parser.add_argument('--data_path', type=str, default='../data')
parser.add_argument('--n_pts', type=int, default=2048, help='number of points per shape')
parser.add_argument('--normalize', action='store_true', help='normalize points')
parser.add_argument('--epochs', type=int, default=350)
parser.add_argument('--batch_size', type=int, default=24)
parser.add_argument('--test_batch_size', type=int, default=32)
parser.add_argument('--optim_type', type=str, default='adam', choices=['sgd', 'adam'], help='sgd or adam')
parser.add_argument('--sched_type', type=str, default='cos', choices=['cos', 'step'], help='scheduler type')
parser.add_argument('--step_size', type=int, default=40, help='step size for learning rate decay')
parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decay ratio')
parser.add_argument('--lr', type=float, default=0.005, help='default set to 0.003')
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--model_type', type=str, default='insiou', choices=['insiou', 'clsiou', 'acc'], help='best model type')
parser.add_argument('--resume', action='store_true', help='resume checkpoint')
parser.add_argument('--eval', action='store_true', help='flag to model evaluation')

parser.add_argument('--log_path', type=str, default='log', help='log path')
parser.add_argument('--img_path', type=str, default='image', help='image path')
parser.add_argument('--out_path', type=str, default='image', help='output path')
parser.add_argument('--ck_path', type=str, default='checkpoint', help='checkpoint path')

parser.add_argument('--log_name', type=str, default='train.log', help='log name')

parser.add_argument('--visualize', action='store_true', help='visualize segmentation output')

parser.add_argument('--comment', type=str, default=None, help='comment for experiments')

# for distributed parallel data parallel training
parser.add_argument('--nodes', type=int, default=1, help='number of nodes')
parser.add_argument('--gpus', type=int, default=2, help='number of gpus per node')
parser.add_argument('--n_rank', type=int, default=0, help='rank of each node: 0 for 1st node, 1 for 2nd node, ...')
parser.add_argument('--world_size', type=int, default=1, help='word size')
parser.add_argument('--sync_bn', action='store_true', help='synchronization of batchnorm statistics')

parser.add_argument('--dropout', type=float, nargs='+', default=[0.1, 0.4])

opts = parser.parse_args()