import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.nn.functional import relu
import itertools
from torch.autograd import Variable
import re
import os
from shutil import copyfile
import numpy as np
import torchvision
from collections import OrderedDict

NUM_OF_MOCK_IMGS = np.random.randint(2, 16)
IMAGE_DIMS = torch.Tensor([NUM_OF_MOCK_IMGS, 3, 128, 128])
MOCK_IMAGE = Variable(torch.rand(tuple(IMAGE_DIMS)))
IMAGE_LENGTH = IMAGE_DIMS.data[2]
IMAGE_DEPTH = IMAGE_DIMS.data[1]

KERNEL_SIZE = 2
STRIDE_SIZE = 2

NUM_ENCODER_CHANNELS = 64
NUM_Z_CHANNELS = 50
NUM_GEN_CHANNELS = 1024

import random

NUM_AGES = 10
MOCK_AGES = -torch.ones(NUM_OF_MOCK_IMGS, NUM_AGES)
NUM_GENDERS = 2
MOCK_GENDERS = -torch.ones(NUM_OF_MOCK_IMGS, NUM_GENDERS)
for i in range(NUM_OF_MOCK_IMGS):
    MOCK_GENDERS[i][random.getrandbits(1)] *= -1  # random hot gender
    MOCK_AGES[i][random.randint(0, NUM_AGES - 1)] *= -1  # random hot age

MOCK_LABELS = torch.cat((MOCK_AGES, MOCK_GENDERS), 1)

image_value_range = (-1, 1)
num_categories = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        num_conv_layers = int(torch.log2(IMAGE_LENGTH)) - int(KERNEL_SIZE / 2)

        self.conv_layers = nn.ModuleList()

        for i in range(1, num_conv_layers + 1):
            self.conv_layers.add_module('e_conv_%d' % i, nn.Sequential(*[
                    nn.Conv2d(
                        in_channels=(NUM_ENCODER_CHANNELS * 2**(i-2)) if i > 1 else int(IMAGE_DEPTH),
                        out_channels=NUM_ENCODER_CHANNELS * 2**(i-1),
                        kernel_size=KERNEL_SIZE,
                        stride=STRIDE_SIZE
                    ),
                    nn.ReLU()
            ]))

        self.fc_layer = nn.Sequential(OrderedDict([
            ('e_fc_1', nn.Linear(
                in_features=NUM_ENCODER_CHANNELS * int(IMAGE_LENGTH**2) // int(2**(num_conv_layers+1)),
                out_features=NUM_Z_CHANNELS
            )),
            ('tanh_1', nn.Tanh())
        ]))

    def forward(self, face):
        out = face
        for conv_layer in self.conv_layers:
            out = conv_layer(out)
        out = self.fc_layer(out.view(out.size(0), -1))  # flatten tensor (reshape)
        return out


class DiscriminatorZ(nn.Module):
    def __init__(self):
        super(DiscriminatorZ, self).__init__()
        dims = (NUM_Z_CHANNELS, NUM_ENCODER_CHANNELS, NUM_ENCODER_CHANNELS // 2, NUM_ENCODER_CHANNELS // 4)
        self.layers = nn.ModuleList()

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
        num_deconv_layers = int(torch.log2(IMAGE_LENGTH)) - int(KERNEL_SIZE / 2) -1  # TODO
        mini_size = 8
        self.fc = nn.Sequential(
            nn.Linear(NUM_Z_CHANNELS + NUM_AGES + NUM_GENDERS, NUM_GEN_CHANNELS * mini_size**2),
            nn.ReLU()
        )
        # need to reshape now to ?,1024,8,8

        self.conv_layers = nn.ModuleList()

        for i in range(1, num_deconv_layers + 1):
            self.conv_layers.add_module('g_deconv_%d' % i, nn.Sequential(*[
                nn.ConvTranspose2d(
                    in_channels=int(NUM_GEN_CHANNELS // (2 ** (i - 1))),
                    out_channels=int(NUM_GEN_CHANNELS // (2 ** (i - 0))) if i < num_deconv_layers else int(IMAGE_DEPTH),
                    kernel_size=KERNEL_SIZE if i < num_deconv_layers else 1,
                    stride=STRIDE_SIZE if i < num_deconv_layers else 1
                ),
                nn.ReLU()
            ]))

    def forward(self, z, age, gender):
        z_l = torch.cat((z, age, gender), 1)
        out = self.fc(z_l)
        out = out.view(out.size(0), 1024, 8, 8)  # TODO - replace hardcoded
        for conv_layer in self.conv_layers:
            out = conv_layer(out)

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

    def to(self, device):
        for subnet in self.subnets:
            subnet.to(device=device)


utkface_original_image_format = re.compile('^(\d+)_(\d+)_\d+_(\d+)\.jpg\.chip\.jpg$')

MALE = 0
FEMALE = 1

from collections import namedtuple

class Label(namedtuple('Label', ('age', 'gender'))):
    def __init__(self, age, gender):
        super(Label, self).__init__()
        _age = self.age - 1
        if _age < 20:
            self.age_group = max(_age // 5, 0)  # first 4 age groups are for kids <= 20, 5 years intervals
        else:
            self.age_group = min(4 + (_age - 20) // 10, NUM_AGES - 1)  # last (6?) age groups are for adults > 20, 10 years intervals

    def to_str(self):
        return '%d_%d' % (self.age_group, self.gender)

    def to_tensor(self):
        age_tensor = -torch.ones(NUM_AGES)
        age_tensor[self.age_group] *= -1
        gender_tensor = -torch.ones(NUM_GENDERS)
        gender_tensor[self.gender] *= -1
        result = torch.cat((age_tensor, gender_tensor), 0)
        result = result.to(device=device)
        return result


def sort_to_classes(root, print_cycle=np.inf):
    # Example UTKFace cropped and aligned image file format: [age]_[gender]_[race]_[date&time].jpg.chip.jpg
    # Should be 23613 images, use print_cycle >= 1000
    # Make sure you have > 100 MB free space

    def log(text):
        print('[UTKFace dset labeler] ' + text)

    log('Starting labeling process...')
    files = [f for f in os.listdir(root) if os.path.isfile(os.path.join(root, f)) and utkface_original_image_format.match(f)]
    if not files:
        raise FileNotFoundError('No image files in '+root)
    copied_count = 0
    sorted_folder = os.path.join(root, '..', 'labeled')
    if not os.path.isdir(sorted_folder):
        os.mkdir(sorted_folder)

    for f in files:
        srcfile = os.path.join(root, f)
        age, gender, dtime = matcher.groups()
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
    ret = lambda: torchvision.datasets.ImageFolder(os.path.join(root, 'labeled'), transform=transforms.ToTensor())
    try:
        return ret()
    except (RuntimeError, FileNotFoundError):
        sort_to_classes(os.path.join(root, 'unlabeled'), print_cycle=1000)
        return ret()


train_dataset = get_utkface_dataset(os.path.join('.', 'data', 'UTKFace'))
batch_size = 100
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)


# Turn image into a batch of size 1, 128x128, RGB
# MOCK_IMAGE.unsqueeze_(0)
MOCK_IMAGE = MOCK_IMAGE.to(device)
MOCK_AGES = MOCK_AGES.to(device)
MOCK_GENDERS = MOCK_GENDERS.to(device)

net = Net()
net.to(device=device)

z = net.E(MOCK_IMAGE)
dz = net.Dz(z)
output = net.G(z, MOCK_AGES, MOCK_GENDERS)

print(output.size())
print(device)
