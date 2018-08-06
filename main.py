import model
import consts
import logging
import os
import re
import numpy as np
import argparse
import sys
import random
import datetime
import torch
import utils
from torchvision.datasets.folder import pil_loader

assert sys.version_info >= (3, 6),\
    "This script requires Python >= 3.6"  # TODO 3.7?
assert tuple(int(ver_num) for ver_num in torch.__version__.split('.')) >= (0, 4, 0),\
    "This script requires PyTorch >= 0.4.0"  # TODO 0.4.1?

def str_to_gender(s):
    s = str(s).lower()
    if s in ('m', 'man', '0'):
        return 0
    elif s in ('f', 'female', '1'):
        return 1
    else:
        raise Exception()

if 'net' not in globals() and False:  # for interactive execution in PyCharm
    net = model.Net()
    net.to(device=consts.device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AgeProgression on PyTorch.')
    parser.add_argument('--mode', choices=['train', 'test'], default='train')

    # train params
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--bs', '--batch-size', dest='batch_size', default=64, type=int)
    parser.add_argument('--wd', '--weight-decay', dest='weight_decay', default=1e-5, type=float)
    parser.add_argument('--lr', '--learning-rate', dest='learning_rate', default=2e-4, type=float)
    parser.add_argument('--b1', '--beta1', dest='b1', default=0.9, type=float)
    parser.add_argument('--b2', '--beta2', dest='b2', default=0.999, type=float)

    # test params
    parser.add_argument('--age', required=False, type=int)
    parser.add_argument('--gender', required=False, type=str_to_gender)

    # shared params
    parser.add_argument('--cuda', default=True, type=bool)
    parser.add_argument('--load', required=False, default=None)  # for pre-training or testing
    parser.add_argument('--input', '-i', dest='input', help='Training dataset path or testing image path')
    parser.add_argument('--output', '-o', dest='output', default='')  # for checkpoints or test results


    args = parser.parse_args()

    try:
        os.remove(r'results/log_results.log')
    except:
        pass
    logging.basicConfig(filename=r'results/log_results.log', level=logging.DEBUG)

    net = model.Net()
    if args.cuda:
        net.cuda()
    else:
        net.cpu()

    if args.mode == 'train':

        if args.load is not None:
            net.load(args.load)

        results_dest = args.output or utils.default_train_results_dir()

        net.teach(
            utkface_path=args.input,
            batch_size=args.batch_size,
            betas=(args.b1, args.b2),
            epochs=args.epochs,
            weight_decay=args.weight_decay,
            learning_rate=args.learning_rate,
            name=results_dest,
        )

    elif args.mode == 'test':

        if args.load is None:
            raise RuntimeError("Must provide path of trained models")

        results_dest = args.output or utils.default_test_results_dir()

        img = utils.pil_to_model_tensor_transform(pil_loader(args.input))
        net.test_single(img_tensor=img, age=args.age, gender=args.gender, target=results_dest)

