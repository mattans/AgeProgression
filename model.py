import consts
from utils import *
import os
from shutil import copyfile
import numpy as np
from collections import OrderedDict, namedtuple
from torchvision.utils import save_image
from torchvision import datasets
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import datetime
from scipy.misc import imsave as ims
import torchvision
from torchvision.datasets import ImageFolder
# from torch.nn.functional import relu

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        num_conv_layers = int(torch.log2(consts.IMAGE_LENGTH)) - int(consts.KERNEL_SIZE / 2)

        self.conv_layers = nn.ModuleList()

        for i in range(1, num_conv_layers + 1):
            not_input_layer = i > 1
            self.conv_layers.add_module('e_conv_%d' % i, nn.Sequential(*[
                    nn.Conv2d(
                        in_channels=(consts.NUM_ENCODER_CHANNELS * 2**(i-2)) if not_input_layer else int(consts.IMAGE_DEPTH),
                        out_channels=consts.NUM_ENCODER_CHANNELS * 2**(i-1),
                        kernel_size=consts.KERNEL_SIZE,
                        stride=consts.STRIDE_SIZE,
                        padding=2
                    ),
                    nn.ReLU()
            ]))

        self.fc_layer = nn.Sequential(OrderedDict([
            ('e_fc_1', nn.Linear(
                in_features=consts.NUM_ENCODER_CHANNELS * int(consts.IMAGE_LENGTH**2) // int(2**(num_conv_layers+1)),
                out_features=consts.NUM_Z_CHANNELS
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
        dims = (consts.NUM_Z_CHANNELS, consts.NUM_ENCODER_CHANNELS, consts.NUM_ENCODER_CHANNELS // 2, consts.NUM_ENCODER_CHANNELS // 4)
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
        num_deconv_layers = int(torch.log2(consts.IMAGE_LENGTH)) - int(consts.KERNEL_SIZE / 2) # TODO
        mini_size = 8
        self.fc = nn.Sequential(
            nn.Linear(consts.NUM_Z_CHANNELS + consts.NUM_AGES + consts.NUM_GENDERS, consts.NUM_GEN_CHANNELS * mini_size**2),
            nn.ReLU()
        )
        # need to reshape now to ?,1024,8,8

        self.deconv_layers = nn.ModuleList()

        for i in range(1, num_deconv_layers + 1):
            not_output_layer = i < num_deconv_layers
            self.deconv_layers.add_module('g_deconv_%d' % i, nn.Sequential(*[
                nn.ConvTranspose2d(
                    in_channels=int(consts.NUM_GEN_CHANNELS // (2 ** (i - 1))),
                    out_channels=int(consts.NUM_GEN_CHANNELS // (2 ** i)) if not_output_layer else 3,
                    kernel_size=2 if not_output_layer else 1,
                    stride=2 if not_output_layer else 1,
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
            out = self._uncompress(out)
            if debug:
                print("G FC output size: " + str(out.size()))
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

    def train(self, utkface_path, batch_size=50, epochs=1, weight_decay=0.0, lr=1e-3, size_average=False):
        print("DEBUG: starting train")
        train_dataset = get_utkface_dataset(utkface_path)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}
        eg_optimizer = Adam(list(self.E.parameters()) + list(self.G.parameters()), weight_decay=0.5, lr=lr)
        criterion = nn.L1Loss(size_average=size_average)  # L2 loss

        test_loader = DataLoader(get_utkface_dataset(utkface_path), batch_size=batch_size, shuffle=True)
        test_images = None
        test_labels = None
        for i, (images, labels) in enumerate(test_loader):
            #data = images.to(consts.device)
            #recon_batch, mu, logvar = model(data)
            test_images = images.to(device=consts.device)
            test_labels = torch.stack(
                [str_to_tensor(idx_to_class[l]).to(device=consts.device) for l in list(labels.numpy())])
            test_labels = test_labels.to(device=consts.device)
            # if i == 0:
            #             #     n = min(data.size(0), 8)
            #             #     comparison = torch.cat([data[:n],
            #             #                             recon_batch.view(batch_size, 1, 28, 28)[:n]])
            #             #     save_image(data,
            #             #                'results/base.png', nrow=n)
            break

        torchvision.utils.save_image(test_images, "./results/base.png", nrow=8)
        epoch_losses = []
        for epoch in range(1, epochs + 1):
            epoch_loss = 0
            for i, (images, labels) in enumerate(train_loader, 1):
                images = images.to(device=consts.device)
                labels = torch.stack([str_to_tensor(idx_to_class[l]).to(device=consts.device) for l in list(labels.numpy())])
                labels = labels.to(device=consts.device)

                z = self.E(images)
                z_l = torch.cat((z, labels), 1)
                generated = self.G(z_l)

                loss = criterion(generated, images)

                eg_optimizer.zero_grad()
                loss.backward()
                eg_optimizer.step()

                now = datetime.datetime.now()

                print(f"TRAIN:[{now.hour:d}:{now.minute:d}] [Epoch {epoch:d}, Batch {i:d}] Loss: {loss.item():f}")
                epoch_loss += loss.item()
                # break
            if (True):
                #####test#####
                print("DEBUG: ##############test###############")
                z = self.E(test_images)
                #z_l = torch.cat((z, test_labels), 1)
                generated = self.G(z)
                test_loss = criterion(generated, test_images)
                now = datetime.datetime.now()
                print(f"TEST: [{now.hour:d}:{now.minute:d}] [Epoch {epoch:d}, Loss: {test_loss.item():f}")
                torchvision.utils.save_image(generated, 'results/img_' + str(epoch) + '.png', nrow=8)
                # save_image(generated.view(batch_size, 3, 128, 128),
                #            'results/img_b_' + str(epoch) + '.png')
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










