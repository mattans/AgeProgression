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
import logging
# from torch.nn.functional import relu
from torch.utils.data.sampler import SubsetRandomSampler

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

class DiscriminatorImg(nn.Module):
    def __init__(self):
        super(DiscriminatorImg, self).__init__()
        dims = (3, 16, 32, 64, 128)
        self.conv_layers = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:]), 1):
            self.layers.add_module(
                'dimg_conv_%d' % i,
                nn.Sequential(
                    nn.Conv2d(in_dim, out_dim),
                    nn.BatchNorm1d(out_dim),
                    nn.ReLU()
                )
            )

        self.fc_1 = nn.Sequential(
                nn.Linear(128*8*8, 1024),
                nn.ReLU()
        )

        self.fc_2 = nn.Sequential(
                nn.Linear(1024, 1),
                # nn.Sigmoid()
        )

    def forward(self, img, label):
        out = img
        out = self.conv_layers[0]
        out = torch.cat((out, label.to_tensor(equalize_weight=True)), 0)
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

    def _decompress(self, x):
        return x.view(x.size(0), 1024, 8, 8)  # TODO - replace hardcoded

    def forward(self, z, age=None, gender=None, debug=False, debug_fc=False, debug_deconv=[]):
        out = z
        if age is not None and gender is not None:
            label = Label(age, gender).to_tensor()\
                if (isinstance(age, int) and isinstance(gender, int))\
                else torch.cat((age, gender), 1)
            out = torch.cat((out, label), 1)  # z_l
        if (not debug) or (debug and debug_fc):
            out = self.fc(out)
            out = self._decompress(out)
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
        self.fitting_warning = False

    def __call__(self, x):
        z = self.E(x)
        z_disc = self.Dz(z)
        return z_disc

    def __repr__(self):
        return os.linesep.join([repr(subnet) for subnet in self.subnets])

    def train(
            self,
            utkface_path,
            batch_size=64,
            epochs=1,
            weight_decay=1e-5,
            learning_rate=2e-4,
            betas=(0.9, 0.999),
            name=default_results_dir(),
            valid_size=consts.BATCH_SIZE,
    ):

        train_dataset = get_utkface_dataset(utkface_path)
        valid_dataset = get_utkface_dataset(utkface_path)
        dset_size = len(train_dataset)
        indices = list(range(dset_size))
        # split = int(np.floor(valid_size * dset_size))
        split = int(np.floor(valid_size))
        # np.random.seed(random_seed)
        np.random.shuffle(indices)
        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=train_sampler)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, sampler=valid_sampler)
        idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}

        validate_images = None
        validate_labels = None
        for ii, (images, labels) in enumerate(valid_loader, 1):
            validate_images = images.to(device=consts.device)
            labels = torch.stack(
                [str_to_tensor(idx_to_class[l]).to(device=consts.device) for l in list(labels.numpy())])
            validate_labels = labels.to(device=consts.device)
        joined_image = one_sided(validate_images)

        torchvision.utils.save_image(joined_image, "./results/base.png")#, nrow=8)

        eg_optimizer, eg_criterion = optimizer_and_criterion(nn.L1Loss, Adam, self.E, self.G, weight_decay=weight_decay, betas=betas, lr=learning_rate)
        z_optimizer, z_criterion = optimizer_and_criterion(nn.BCEWithLogitsLoss, Adam, self.Dz, weight_decay=weight_decay, betas=betas, lr=learning_rate)

        #  TODO - write a txt file with all arguments to results folder

        epoch_losses = []
        epoch_losses_valid = []
        loss_tracker = LossTracker()
        z_prior = 255 * torch.rand(batch_size, consts.NUM_Z_CHANNELS)
        d_z_prior = self.Dz(z_prior.to(device=consts.device))
        save_count = 0
        for epoch in range(1, epochs + 1):
            epoch_loss = 0
            epoch_loss_valid = 0
            for i, (images, labels) in enumerate(train_loader, 1):

                for subnet in self.subnets:
                    subnet.train()  # move to train mode

                images = images.to(device=consts.device)
                labels = torch.stack([str_to_tensor(idx_to_class[l]).to(device=consts.device)
                                      for l in list(labels.numpy())])
                labels = labels.to(device=consts.device)
                print ("DEBUG: iteration: "+str(i)+" images shape: "+str(images.shape))
                z = self.E(images)
                if(z.shape != z_prior.shape):
                    z_prior = 255 * torch.rand(z.shape[0], consts.NUM_Z_CHANNELS)
                    d_z_prior = self.Dz(z_prior.to(device=consts.device))
                z_l = torch.cat((z, labels), 1)
                generated = self.G(z_l)
                eg_loss = eg_criterion(generated, images)

                reg_loss = 0 * (
                        torch.sum(torch.abs(generated[:, :, :, :-1] - generated[:, :, :, 1:])) +
                        torch.sum(torch.abs(generated[:, :, :-1, :] - generated[:, :, 1:, :]))
                ) / batch_size  # TO DO - ADD TOTAL VARIANCE LOSS

                d_z = self.Dz(z)
                dz_loss = z_criterion(d_z_prior, d_z)
                eg_optimizer.zero_grad()
                z_optimizer.zero_grad()
                loss = eg_loss + reg_loss + dz_loss
                loss.backward(retain_graph=True)
                eg_optimizer.step()
                z_optimizer.step()
                now = datetime.datetime.now()

                epoch_loss += loss.item()
                if save_count % 500 == 0:
                    logging.info('[{h}:{m}[Epoch {e}, i: {c}] Loss: {t}'.format(h=now.hour, m=now.minute, e=epoch, c=i,
                                                                                t=loss.item()))
                    print(f"[{now.hour:d}:{now.minute:d}] [Epoch {epoch:d}, i {i:d}] Loss: {loss.item():f}")
                    cp_path = self.save(name)
                    # joined_image = one_sided(torch.cat((images, generated), 0))
                    # save_image(joined_image, os.path.join(cp_path, 'reconstruct.png'))
                save_count += 1
            epoch_losses += [epoch_loss / i]

            with torch.no_grad():  # validation

                for subnet in self.subnets:
                    subnet.eval()  # move to eval mode

                z = self.E(validate_images)
                z_l = torch.cat((z, validate_labels), 1)
                generated = self.G(z_l)
                loss = nn.functional.l1_loss(validate_images, generated)
                joined_image = one_sided(generated)
                #torchvision.utils.save_image(generated, 'results/img_' + str(epoch) + '.png', nrow=8)
                torchvision.utils.save_image(joined_image, 'results/onesided_' + str(epoch) + '.png', nrow=8)
                epoch_loss_valid += loss.item()
            epoch_losses_valid += [epoch_loss_valid/ii]

            loss_tracker.append(epoch_loss / i, epoch_loss_valid / ii, cp_path)
            try:
                logging.info('[{h}:{m}[Epoch {e}] Train Loss: {t} Vlidation Loss: {v}'.format(h=now.hour, m=now.minute,
                                                                                              e=epoch, t=epoch_losses[-1],
                                                                                              v=epoch_losses_valid[-1]))
                print(f"[{now.hour:d}:{now.minute:d}] [Epoch {epoch:d}] Train Loss: {epoch_losses[-1]:f} Validation Loss: "
                      f"{epoch_losses_valid[-1]:f}")
            except IndexError as e:
                logging.error('[{h}:{m}' + str(e))
                logging.error('[{h}:{m} epoch_losses: ' + str(epoch_losses))
                logging.error('[{h}:{m} epoch_losses_valid: ' + str(epoch_losses_valid))
                print(e)
                print("epoch_losses: " + str(epoch_losses))
                print("epoch_losses_valid: " + str(epoch_losses_valid))



    def to(self, device):
        for subnet in self.subnets:
            subnet.to(device=device)

    def cpu(self):
        for subnet in self.subnets:
            subnet.cpu()

    def cuda(self):
        for subnet in self.subnets:
            subnet.cuda()

    def save(self, path):
        if not os.path.isdir(path):
            os.mkdir(path)
        path = os.path.join(path, datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
        if not os.path.isdir(path):
            os.mkdir(path)
        e_sd = self.E.state_dict()
        g_sd = self.G.state_dict()
        # TOOO - loop over self.subnets
        torch.save(e_sd, os.path.join(path, "E.dat"))
        torch.save(g_sd, os.path.join(path, "G.dat"))
        print("Saved to " + path)
        return path

    def load(self, path):
        e_sd = torch.load(os.path.join(path, "E.dat"))
        g_sd = torch.load(os.path.join(path, "G.dat"))
        self.E.load_state_dict(e_sd)
        self.G.load_state_dict(g_sd)
        print("Loaded from " + path)

