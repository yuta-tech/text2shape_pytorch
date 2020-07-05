import torch
from torch.utils.data import DataLoader
import numpy as np
import argparse

import utils.dataset as dataset

from models.config import cfg, cfg_from_list

from trainer import Trainer

def set_seed():
    # set random seed, default is 0
    # cudnn
    torch.backends.cudnn.deterministic = True
    # random
    seed = cfg.GAN.SEED
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

def arg_parse(parser):
    # parse args
    parser.add_argument('--summary', type=bool, default=False, help='if summarize model -> True')
    
    args = parser.parse_args()
    
    return args

# def fix_config(args):
#     if args.ngf is not None:
#         cfg_from_list(['GAN.GF_DIM', args.ngf])
#     if args.ndf is not None:
#         cfg_from_list(['GAN.DF_DIM', args.ndf])
#     if args.ncf is not None:
#         cfg_from_list(['GAN.CONDITION_DIM', args.ncf])

def get_dataloader(mode='train', batch_size=8, resplit=False):
    shapenet = dataset.ShapeNetDataset(mode=mode, resplit=resplit)
    return DataLoader(shapenet, batch_size, shuffle=True)

if __name__ == '__main__':
    # argparse
    parser = argparse.ArgumentParser()
    args = arg_parse(parser)
#     fix_config(args)
    # device selection
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed()
    train_dataloader = get_dataloader(mode='train', batch_size=cfg.TRAIN.BATCH_SIZE)
    
    t2s_trainer = Trainer(train_dataloader, device)
    t2s_trainer.train(summary=args.summary)
    
#     arg = arg_parse()
