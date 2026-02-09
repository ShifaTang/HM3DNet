import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--epoch',       type=int,   default=200,   help='epoch number')
parser.add_argument('--lr',          type=float, default=1e-4,  help='learning rate')
parser.add_argument('--batchsize',   type=int,   default=16,    help='training batch size')
parser.add_argument('--trainsize',   type=int,   default=256,   help='training dataset size')
parser.add_argument('--clip',        type=float, default=0.5,   help='gradient clipping margin')
parser.add_argument('--lw',          type=float, default=0.001, help='weight')
parser.add_argument('--decay_rate',  type=float, default=0.85,   help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int,   default=20,    help='every n epochs decay learning rate')
parser.add_argument('--load',        type=str,   default=None,  help='train from checkpoints')
parser.add_argument('--gpu_id',      type=str,   default='0',   help='train use gpu')

parser.add_argument('--rgb_label_root',      type=str, default='/home/accv/tsf/new-imag2.0/rgb/',           help='the training rgb images root')
parser.add_argument('--depth_label_root',    type=str, default='/home/accv/tsf/new-imag2.0/d/',         help='the training depth images root')
parser.add_argument('--gt_label_root',       type=str, default='/home/accv/tsf/new-imag2.0/gt/',            help='the training gt images root')

parser.add_argument('--test_rgb_root',        type=str, default='/home/accv/tsf/new-imag2.0/rgb-test/',      help='the test rgb images root')
parser.add_argument('--test_depth_root',      type=str, default='/home/accv/tsf/new-imag2.0/d-test/',    help='the test depth images root')
parser.add_argument('--test_gt_root',         type=str, default='/home/accv/tsf/new-imag2.0/gt-test/',       help='the test gt images root')


parser.add_argument('--save_path',           type=str, default='/home/accv/tsf/train_val_checkpoints/',    help='the path to save models and logs')


opt = parser.parse_args()

