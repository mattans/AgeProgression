import model
import consts

import re
import numpy as np
import argparse
import sys
import random
import datetime
import torch
import utils

assert sys.version_info >= (3, 6),\
    "This script requires Python >= 3.6"  # TODO 3.7?
assert tuple(int(ver_num) for ver_num in torch.__version__.split('.')) >= (0, 4, 0),\
    "This script requires PyTorch >= 0.4.0"  # TODO 0.4.1?

if 'net' not in globals() and False:  # for interactive execution in PyCharm
    net = model.Net()
    net.to(device=consts.device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This.')
    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    parser.add_argument('--cuda', default=True, type=bool)

    # train params
    parser.add_argument('--tdset', '--train-dataset', dest='train_dataset', default='./data/UTKFace')
    parser.add_argument('--pretrain', required=False, default=None)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--bs', '--batch-size', dest='batch_size', default=64, type=int)
    parser.add_argument('--wd', '--weight-decay', dest='weight_decay', default=1e-5, type=float)
    parser.add_argument('--lr', '--learning-rate', dest='learning_rate', default=2e-4, type=float)
    parser.add_argument('--b1', '--beta1', dest='b1', default=0.9, type=float)
    parser.add_argument('--b2', '--beta2', dest='b2', default=0.999, type=float)
    parser.add_argument('--resdest', '--results-dest', dest='results_dest', default='')

    args = parser.parse_args()

    net = model.Net()
    if args.cuda:
        net.cuda()
    else:
        net.cpu()

    if args.mode == 'train':

        if args.pretrain is not None:
            net.load(args.pretrain)



        results_dest = args.results_dest or utils.default_results_dir()


        net.teach(
            utkface_path=args.train_dataset,
            batch_size=args.batch_size,
            betas=(args.b1, args.b2),
            epochs=args.epochs,
            weight_decay=args.weight_decay,
            learning_rate=args.learning_rate,
            name=results_dest,
)