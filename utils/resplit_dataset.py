import dataset
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--train_split', type=float, default=0.8, help='train size [0.0-1.0]')
parser.add_argument('--val_split', type=float, default=0.1, help='val size [0.0-1.0]')
parser.add_argument('--seed', type=int, default=0, help='seed value as random state')

args = parser.parse_args()

dataset.resplit_sample_train_val(args.train_split, args.val_split, args.seed)