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
import gc
import torch

gc.collect()

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AgeProgression on PyTorch.')
    parser.add_argument('--mode', choices=['train', 'test'], default='train')

    # train params
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--models-saving', dest='models_saving', choices=['always', 'last', 'never'], default='always', type=str)
    parser.add_argument('--bs', '--batch-size', dest='batch_size', default=64, type=int)
    parser.add_argument('--wd', '--weight-decay', dest='weight_decay', default=1e-5, type=float)
    parser.add_argument('--lr', '--learning-rate', dest='lr', default=2e-4, type=float)
    parser.add_argument('--b1', '--beta1', dest='b1', default=0.5, type=float)
    parser.add_argument('--b2', '--beta2', dest='b2', default=0.999, type=float)
    parser.add_argument('--sp', '--shouldplot', dest='sp', default=False, type=bool)

    # test params
    parser.add_argument('--age', required=False, type=int)
    parser.add_argument('--gender', required=False, type=str_to_gender)

    # shared params
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--load', required=False, default=None, help='Trained models path for pre-training or for testing')
    parser.add_argument(
        '--input',
        '-i',
        default=None,
        help='Training dataset path (default is {}) or testing image path'.format(utils.default_train_results_dir())
    )
    parser.add_argument('--output', '-o', default='')
    args = parser.parse_args()

    try:
        os.remove(r'results/log_results.log')
    except OSError:
        pass
    logging.basicConfig(filename=r'results/log_results.log', level=logging.DEBUG)

    net = model.Net()
    if args.cpu:  # force usage of cpu even if cuda is available (can be faster for testing)
        net.cpu()

    if args.mode == 'train':

        betas = (args.b1, args.b2) if args.load is None else None
        weight_decay = args.weight_decay if args.load is None else None
        lr = args.lr if args.load is None else None

        if args.load is not None:
            net.load(args.load)
            print("Loading pre-trained models from {}".format(args.load))

        data_src = args.input or os.path.join('.', 'data', 'UTKFace')
        print("Data folder is {}".format(data_src))
        results_dest = args.output or utils.default_train_results_dir()
        print("Results folder is {}".format(results_dest))

        #Create info text file besides the epochs directories

        path_str = os.path.join(results_dest, 'results')
        try:
            os.remove(os.path.join(path_str, 'log_results.log'))
        except:
            if not os.path.exists(path_str):
                os.makedirs(path_str)
        with open(os.path.join(results_dest , 'session_arguments.txt'), 'a') as info_file:
            info_file.write(' '.join(sys.argv))


        logging.basicConfig(filename=os.path.join(path_str , 'log_results.log'), level=logging.DEBUG)


        net.teach(
            utkface_path=data_src,
            batch_size=args.batch_size,
            betas=betas,
            epochs=args.epochs,
            weight_decay=weight_decay,
            lr=lr,
            should_plot=args.sp,
            where_to_save=results_dest,
            models_saving=args.models_saving
        )

    elif args.mode == 'test':

        if args.load is None:
            raise RuntimeError("Must provide path of trained models")

        net.load(args.load)

        results_dest = args.output or utils.default_test_results_dir()
        try:
            os.makedirs(results_dest)
        except:
            pass

        img = utils.pil_to_model_tensor_transform(pil_loader(args.input)) #Covert path to image (pil_loader)
        if not args.cpu and torch.cuda.is_available():
            img = img.cuda()
        else:
            img = img.cpu()
        net.test_single(img_tensor=img, age=args.age, gender=args.gender, target=results_dest)
