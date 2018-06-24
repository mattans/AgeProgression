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

NUM_OF_MOCK_IMGS = np.random.randint(2,16)
IMAGE_DIMS = torch.Tensor([NUM_OF_MOCK_IMGS, 3, 128, 128])
MOCK_IMAGE = torch.rand(tuple(IMAGE_DIMS))
IMAGE_LENGTH = IMAGE_DIMS.data[2]
IMAGE_DEPTH = IMAGE_DIMS.data[1]
STEP_SIZE = 2  # kernel and stride
NUM_ENCODER_CHANNELS = 64
NUM_Z_CHANNELS = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        num_conv_layers = int(torch.log2(IMAGE_LENGTH))
        conv_layers = []

        for i in range(num_conv_layers):
            conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=(NUM_ENCODER_CHANNELS * (2**(i-1))) if i > 0 else int(IMAGE_DEPTH),
                        out_channels=NUM_ENCODER_CHANNELS * (2**i),
                        kernel_size=STEP_SIZE,
                        stride=STEP_SIZE
                    ),
                    nn.ReLU()
                )
            )

        self.convs = nn.Sequential(*conv_layers)
        self.fc = nn.Linear(
            in_features=NUM_ENCODER_CHANNELS * (2**(num_conv_layers-1)),
            out_features=NUM_Z_CHANNELS
        )

    def forward(self, input_face):
        out = input_face
        out = self.convs(out)
        out = out.view(out.size(0), -1) # flatten tensor
        z = self.fc(out)
        return z


class DiscriminatorZ(nn.Module):
    def __init__(self):
        super(DiscriminatorZ, self).__init__()
        dims = (NUM_Z_CHANNELS, NUM_ENCODER_CHANNELS, NUM_ENCODER_CHANNELS // 2, NUM_ENCODER_CHANNELS // 4)
        layers = []

        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(
                nn.Sequential(nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU())
            )

        layers.append(
            nn.Sequential(nn.Linear(out_dim, 1), nn.Sigmoid())
        )

        self.layers = nn.Sequential(*layers)

    def forward(self, z):
        out = self.layers(z)
        return out


class Net(object):
    def __init__(self):
        self.E = Encoder()
        self.Dz = DiscriminatorZ()
        self.subnets = (self.E, self.Dz)

    def __call__(self, x):
        z = self.E(x)
        z_disc = self.Dz(z)
        return z_disc

    def to(self, device):
        for subnet in self.subnets:
            subnet.to(device=device)


def sort_to_classes(root, print_cycle=np.inf):
    # Example UTKFace cropped and aligned image file format: [age]_[gender]_[race]_[date&time].jpg.chip.jpg
    # Should be 23613 images, use print_cycle >= 1000
    utkface_original_image_format = re.compile('^(\d+)_\d+_\d+_(\d+)\.jpg\.chip\.jpg$')
    files = iter([f for f in os.listdir(root) if os.path.isfile(os.path.join(root, f))])
    copied_count = 0
    sorted_folder = os.path.join(root, 'sorted')
    if not os.path.isdir(sorted_folder):
        os.mkdir(sorted_folder)

    for f in files:
        matcher = utkface_original_image_format.match(f)
        if matcher is not None:
            age, dtime = matcher.groups()
            folder = os.path.join(sorted_folder, str(int(int(age) / 5)))
            dst = os.path.join(folder, dtime+'.jpg')
            if os.path.isfile(dst):
                continue
            if not os.path.isdir(folder):
                os.mkdir(folder)
            src = os.path.join(root, f)
            copyfile(src, dst)
            copied_count += 1
            if copied_count % print_cycle == 0:
                print('Copied %d files.' % copied_count)


def get_utkface_dataset(root):
    ret = lambda: torchvision.datasets.ImageFolder(os.path.join(root, 'sorted'))
    try:
        return ret()
    except (RuntimeError, FileNotFoundError):
        sort_to_classes(root, print_cycle=1000)
        return ret()


train_dataset = get_utkface_dataset('./data/UTKFace')
batch_size = 100
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)


# Turn image into a batch of size 1, 128x128, RGB
# MOCK_IMAGE.unsqueeze_(0)
MOCK_IMAGE = MOCK_IMAGE.to(device)

net = Net()
net.to(device=device)
print(net(MOCK_IMAGE))
print(device)
