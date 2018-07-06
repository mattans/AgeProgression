import itertools
import re
import os
import sys
from shutil import copyfile
import numpy as np
from collections import OrderedDict, namedtuple
import datetime
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
# from torch.nn.functional import relu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_OF_MOCK_IMGS = np.random.randint(2, 16)
IMAGE_DIMS = torch.Tensor([NUM_OF_MOCK_IMGS, 3, 128, 128])
MOCK_IMAGES = torch.rand(tuple(IMAGE_DIMS))
IMAGE_LENGTH = IMAGE_DIMS.data[2]
IMAGE_DEPTH = IMAGE_DIMS.data[1]

KERNEL_SIZE = 5
STRIDE_SIZE = 2

NUM_ENCODER_CHANNELS = 64
NUM_Z_CHANNELS = 50
NUM_GEN_CHANNELS = 1024


NUM_AGES = 10
MOCK_AGES = -torch.ones(NUM_OF_MOCK_IMGS, NUM_AGES)
NUM_GENDERS = 2
MOCK_GENDERS = -torch.ones(NUM_OF_MOCK_IMGS, NUM_GENDERS)
MOCK_IMAGES = MOCK_IMAGES.to(device)
MOCK_AGES = MOCK_AGES.to(device)
MOCK_GENDERS = MOCK_GENDERS.to(device)

for i in range(NUM_OF_MOCK_IMGS):
    MOCK_GENDERS[i][random.getrandbits(1)] *= -1  # random hot gender
    MOCK_AGES[i][random.randint(0, NUM_AGES - 1)] *= -1  # random hot age

MOCK_LABELS = torch.cat((MOCK_AGES, MOCK_GENDERS), 1)

def two_sided(x):
    return 2 * (x - 0.5)


def one_sided(x):
    return (x + 1) / 2

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        num_conv_layers = int(torch.log2(IMAGE_LENGTH)) - int(KERNEL_SIZE / 2)

        self.conv_layers = nn.ModuleList()

        for i in range(1, num_conv_layers + 1):
            not_input_layer = i > 1
            self.conv_layers.add_module('e_conv_%d' % i, nn.Sequential(*[
                    nn.Conv2d(
                        in_channels=(NUM_ENCODER_CHANNELS * 2**(i-2)) if not_input_layer else int(IMAGE_DEPTH),
                        out_channels=NUM_ENCODER_CHANNELS * 2**(i-1),
                        kernel_size=KERNEL_SIZE,
                        stride=STRIDE_SIZE,
                        padding=2
                    ),
                    nn.ReLU()
            ]))

        self.fc_layer = nn.Sequential(OrderedDict([
            ('e_fc_1', nn.Linear(
                in_features=NUM_ENCODER_CHANNELS * int(IMAGE_LENGTH**2) // int(2**(num_conv_layers+1)),
                out_features=NUM_Z_CHANNELS
            )),
            ('tanh_1', nn.Tanh())  # normalize to [-1, 1] range
        ]))

    def _compress(self, x):
        return x.view(x.size(0), -1)

    def forward(self, face):
        out = face
        for conv_layer in self.conv_layers:
            out = conv_layer(out)
        out = self._compress(out)
        out = self.fc_layer(out)  # flatten tensor (reshape)
        return out


class DiscriminatorZ(nn.Module):
    def __init__(self):
        super(DiscriminatorZ, self).__init__()
        dims = (NUM_Z_CHANNELS, NUM_ENCODER_CHANNELS, NUM_ENCODER_CHANNELS // 2, NUM_ENCODER_CHANNELS // 4)
        self.layers = nn.ModuleList()
        i = 0
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:]), 1):
            self.layers.add_module('dz_fc_%d' % i, nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU()
            )
                                   )

        self.layers.add_module('dz_fc_%d' % (i+1), nn.Sequential(
            nn.Linear(out_dim, 1),
            nn.Sigmoid()
        )
                               )

    def forward(self, z):
        out = z
        for layer in self.layers:
            out = layer(out)
        return out


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        num_deconv_layers = int(torch.log2(IMAGE_LENGTH)) - int(KERNEL_SIZE / 2) # TODO
        mini_size = 8
        self.fc = nn.Sequential(
            nn.Linear(NUM_Z_CHANNELS + NUM_AGES + NUM_GENDERS, NUM_GEN_CHANNELS * mini_size**2),
            nn.ReLU()
        )
        # need to reshape now to ?,1024,8,8

        self.deconv_layers = nn.ModuleList()

        for i in range(1, num_deconv_layers + 1):
            not_output_layer = i < num_deconv_layers
            self.deconv_layers.add_module('g_deconv_%d' % i, nn.Sequential(*[
                nn.ConvTranspose2d(
                    in_channels=int(NUM_GEN_CHANNELS // (2 ** (i - 1))),
                    out_channels=int(NUM_GEN_CHANNELS // (2 ** i)) if not_output_layer else 3,
                    kernel_size=KERNEL_SIZE if not_output_layer else 1,
                    stride=STRIDE_SIZE if not_output_layer else 1,
                    padding=2 if not_output_layer else 0,
                    output_padding=1 if not_output_layer else 0
                ),
                nn.ReLU() if not_output_layer else nn.Tanh()
            ]))

    def _uncompress(self, x):
        return x.view(x.size(0), 1024, 8, 8)  # TODO - replace hardcoded

    def forward(self, z, age=None, gender=None, debug=False, debug_fc=False, debug_deconv=[]):
        out = z
        if age is not None and gender is not None:
            label = Label(age, gender).to_tensor()\
                if (isinstance(age, int) and isinstance(gender, int))\
                else torch.cat((age, gender), 1)
            out = torch.cat((out, label), 1)  # z_l
        if (not debug) or (debug and debug_fc):
            out = self.fc(z)
            if debug:
                print("G FC output size: " + str(out.size()))
        out = self._uncompress(out)
        for i, deconv_layer in enumerate(self.deconv_layers, 1):
            if (not debug) or (debug and (i in debug_deconv)):
                out = deconv_layer(out)
                if debug:
                    print("G CONV {} output size: {}".format(i, out.size()))

        return out


class Net(object):
    def __init__(self):
        self.E = Encoder()
        self.Dz = DiscriminatorZ()
        self.G = Generator()
        self.subnets = (self.E, self.Dz, self.G)

    def __call__(self, x):
        z = self.E(x)
        z_disc = self.Dz(z)
        return z_disc

    def __repr__(self):
        return os.linesep.join([repr(subnet) for subnet in self.subnets])

    def train(self, utkface_path, batch_size=50, epochs=1, weight_decay=0.0, lr=1e-4, size_average=False):
        train_dataset = get_utkface_dataset(utkface_path)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}
        eg_optimizer = Adam(list(self.E.parameters()) + list(self.G.parameters()), weight_decay=weight_decay, lr=lr)
        criterion = nn.MSELoss(size_average=size_average)  # L2 loss

        epoch_losses = []
        for epoch in range(1, epochs + 1):
            epoch_loss = 0
            for i, (images, labels) in enumerate(train_loader, 1):
                images = images.to(device=device)
                labels = torch.stack([str_to_tensor(idx_to_class[l]).to(device=device) for l in list(labels.numpy())])
                labels = labels.to(device=device)

                z = self.E(images)
                z_l = torch.cat((z, labels), 1)
                generated = self.G(z_l)

                loss = criterion(generated, images)

                eg_optimizer.zero_grad()
                loss.backward()
                eg_optimizer.step()
                print(f"[{now.hour:d}:{now.minute:d}] [Epoch {epoch:d}, Batch {i:d}] Loss: {loss.item():f}")
                epoch_loss += loss.item()
            epoch_losses += [epoch_loss / i]


    def to(self, device):
        for subnet in self.subnets:
            subnet.to(device=device)

    def cpu(self):
        for subnet in self.subnets:
            subnet.cpu()

    def cuda(self):
        for subnet in self.subnets:
            subnet.cuda()



utkface_original_image_format = re.compile('^(\d+)_(\d+)_\d+_(\d+)\.jpg\.chip\.jpg$')

MALE = 0
FEMALE = 1


def str_to_tensor(text):
    age_group, gender = text.split('.')
    age_tensor = -torch.ones(NUM_AGES)
    age_tensor[int(age_group)] *= -1
    gender_tensor = -torch.ones(NUM_GENDERS)
    gender_tensor[int(gender)] *= -1
    result = torch.cat((age_tensor, gender_tensor), 0)
    result = result.to(device=device)
    return result


class Label(namedtuple('Label', ('age', 'gender'))):
    def __init__(self, age, gender):
        super(Label, self).__init__()
        _age = self.age - 1
        if _age < 20:
            self.age_group = max(_age // 5, 0)  # first 4 age groups are for kids <= 20, 5 years intervals
        else:
            self.age_group = min(4 + (_age - 20) // 10, NUM_AGES - 1)  # last (6?) age groups are for adults > 20, 10 years intervals

    def to_str(self):
        return '%d.%d' % (self.age_group, self.gender)

    def to_tensor(self):
        return str_to_tensor(self.to_str())


def sort_to_classes(root, print_cycle=np.inf):
    # Example UTKFace cropped and aligned image file format: [age]_[gender]_[race]_[date&time].jpg.chip.jpg
    # Should be 23613 images, use print_cycle >= 1000
    # Make sure you have > 100 MB free space

    def log(text):
        print('[UTKFace dset labeler] ' + text)

    log('Starting labeling process...')
    files = [f for f in os.listdir(root) if os.path.isfile(os.path.join(root, f))]
    if not files:
        raise FileNotFoundError('No image files in '+root)
    copied_count = 0
    sorted_folder = os.path.join(root, '..', 'labeled')
    if not os.path.isdir(sorted_folder):
        os.mkdir(sorted_folder)

    for f in files:
        matcher = utkface_original_image_format.match(f)
        if matcher is None:
            continue
        age, gender, dtime = matcher.groups()
        srcfile = os.path.join(root, f)
        label = Label(int(age), int(gender))
        dstfolder = os.path.join(sorted_folder, label.to_str())
        dstfile = os.path.join(dstfolder, dtime+'.jpg')
        if os.path.isfile(dstfile):
            continue
        if not os.path.isdir(dstfolder):
            os.mkdir(dstfolder)
        copyfile(srcfile, dstfile)
        copied_count += 1
        if copied_count % print_cycle == 0:
            log('Copied %d files.' % copied_count)
    log('Finished labeling process.')


def get_utkface_dataset(root):
    ret = lambda: ImageFolder(os.path.join(root, 'labeled'), transform=transforms.Compose([
        transforms.Resize(size=(128, 128)),
        transforms.ToTensor(),
        two_sided  # [0:1] -> [-1:1]
    ]))
    try:
        return ret()
    except (RuntimeError, FileNotFoundError):
        sort_to_classes(os.path.join(root, 'unlabeled'), print_cycle=1000)
        return ret()


if 'net' not in globals():  # for interactive execution in PyCharm
    net = Net()
    net.to(device=device)

    print(device)

    MOCK_TEST = False
    if MOCK_TEST:
        z_mock = net.E(MOCK_IMAGE)
        dz_mock = net.Dz(z_mocl)
        output_mock = net.G(z_mock, MOCK_AGES, MOCK_GENDERS)
        print(output_mock.size())

