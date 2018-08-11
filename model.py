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

        num_conv_layers = 5

        self.conv_layers = nn.ModuleList()

        for i in range(1, num_conv_layers + 1):
            input_layer = i == 1
            self.conv_layers.add_module('e_conv_%d' % i, nn.Sequential(
                nn.Conv2d(
                    in_channels=(consts.NUM_ENCODER_CHANNELS * 2 ** (i - 2)) if not input_layer else int(consts.IMAGE_DEPTH),
                    out_channels=consts.NUM_ENCODER_CHANNELS * 2 ** (i - 1),
                    kernel_size=2,
                    stride=2,
                ),
                nn.ReLU()
            ))

        self.fc_layer = nn.Sequential(OrderedDict([
            ('e_fc_1', nn.Linear(
                in_features=consts.NUM_ENCODER_CHANNELS * int(consts.IMAGE_LENGTH ** 2) // int(
                    2 ** (num_conv_layers + 1)),
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
        dims = (consts.NUM_Z_CHANNELS, consts.NUM_ENCODER_CHANNELS, consts.NUM_ENCODER_CHANNELS // 2,
                consts.NUM_ENCODER_CHANNELS // 4)
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
            nn.Linear(128 * 8 * 8, 1024),
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
        num_deconv_layers = 5
        mini_size = 8
        self.fc = nn.Sequential(
            nn.Linear(consts.NUM_Z_CHANNELS + consts.NUM_AGES + consts.NUM_GENDERS,
                      consts.NUM_GEN_CHANNELS * mini_size ** 2),
            nn.ReLU()
        )
        # need to reshape now to ?,1024,8,8

        self.deconv_layers = nn.ModuleList()

        for i in range(1, num_deconv_layers + 1):
            output_layer = i == num_deconv_layers
            self.deconv_layers.add_module('g_deconv_%d' % i, nn.Sequential(*[
                nn.ConvTranspose2d(
                    in_channels=int(consts.NUM_GEN_CHANNELS // (2 ** (i - 1))),
                    out_channels=int(consts.NUM_GEN_CHANNELS // (2 ** i)) if not output_layer else 3,
                    kernel_size=2 if not output_layer else 1,
                    stride=2 if not output_layer else 1,
                ),
                nn.ReLU() if not output_layer else nn.Tanh()
            ]))

    def _decompress(self, x):
        return x.view(x.size(0), 1024, 8, 8)  # TODO - replace hardcoded

    def forward(self, z, age=None, gender=None):
        out = z
        if age is not None and gender is not None:
            label = Label(age, gender).to_tensor()\
                if (isinstance(age, int) and isinstance(gender, int))\
                else torch.cat((age, gender), 1)
            out = torch.cat((out, label), 1)  # z_l
        out = self.fc(out)
        out = self._decompress(out)
        for i, deconv_layer in enumerate(self.deconv_layers, 1):
                out = deconv_layer(out)
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

    def test_single(self, img_tensor, age, gender, target):
        self.eval()
        batch = img_tensor.repeat(consts.NUM_AGES, 1, 1, 1)  # N x D x H x W
        batch.to(consts.device)
        print(batch.shape, "batch")
        print(img_tensor.shape, "img")
        z = self.E(batch)  # N x Z
        z.to('cuda')
        print(z.device)

        gender_tensor = -torch.ones(consts.NUM_GENDERS)
        gender_tensor[int(gender)] *= -1
        gender_tensor = gender_tensor.repeat(consts.NUM_AGES, 1)  # apply gender on all images

        age_tensor = -torch.ones(consts.NUM_AGES, consts.NUM_AGES)
        for i in range(consts.NUM_AGES):
            age_tensor[i][i] *= -1  # apply the i'th age group on the i'th image

        l = torch.cat((age_tensor, gender_tensor), 1)
        l = l.cuda()
        print(l.device)
        print(l)
        z_l = torch.cat((z, l), 1)
        print(z_l.shape)

        generated = self.G(z_l)
        print(generated.shape, "g")

        joined = torch.cat((img_tensor.unsqueeze(0), generated), 0)

        # TODO - add the original image with the true age caption on it

        save_image(
            tensor=joined,
            filename=os.path.join(target, 'menifa.png'),
            nrow=joined.size(0),
            normalize=True,
            range=(-1, 1),
        )


    def teach(
            self,
            utkface_path,
            batch_size=64,
            epochs=1,
            weight_decay=1e-5,
            learning_rate=2e-4,
            betas=(0.9, 0.999),
            name=default_train_results_dir(),
            valid_size=None,
    ):

        train_dataset = get_utkface_dataset(utkface_path)
        valid_dataset = get_utkface_dataset(utkface_path)
        dset_size = len(train_dataset)
        indices = list(range(dset_size))
        # split = int(np.floor(valid_size * dset_size))
        valid_size = valid_size or batch_size
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
        dz_optimizer, dz_criterion = optimizer_and_criterion(nn.BCEWithLogitsLoss, Adam, self.Dz, weight_decay=weight_decay, betas=betas, lr=learning_rate)

        #  TODO - write a txt file with all arguments to results folder

        loss_tracker = LossTracker()

        save_count = 0
        for epoch in range(1, epochs + 1):
            epoch_loss = 0
            epoch_loss_valid = 0
            for i, (images, labels) in enumerate(train_loader, 1):

                if images.size(0) != batch_size:
                    continue  # tail batch, we can ignore it

                self.train()  # move to train mode

                images = images.to(device=consts.device)
                labels = torch.stack([str_to_tensor(idx_to_class[l]).to(device=consts.device)
                                      for l in list(labels.numpy())])
                labels = labels.to(device=consts.device)
                if(i%50 == 0):
                    print ("DEBUG: iteration: "+str(i)+" images shape: "+str(images.shape))
                z = self.E(images)
                # if(z.shape != z_prior.shape):
                #     z_prior = 2 * torch.rand(z.shape[0], consts.NUM_Z_CHANNELS) - 1
                #     d_z_prior = self.Dz(z_prior.to(device=consts.device))
                z_l = torch.cat((z, labels), 1)
                generated = self.G(z_l)
                eg_loss = eg_criterion(generated, images)

                reg_loss =  (
                        torch.sum(torch.abs(generated[:, :, :, :-1] - generated[:, :, :, 1:])) +
                        torch.sum(torch.abs(generated[:, :, :-1, :] - generated[:, :, 1:, :]))
                ) / batch_size  # TO DO - ADD TOTAL VARIANCE LOSS

                ####D_Z####
                z_prior = 2 * torch.rand(batch_size, consts.NUM_Z_CHANNELS) - 1  # [-1 : 1]
                d_z_prior = self.Dz(z_prior.to(device=consts.device))
                d_z = self.Dz(z)


                dz_loss_prior = dz_criterion(d_z_prior, torch.ones_like(d_z_prior))
                dz_loss = dz_criterion(d_z, torch.zeros_like(d_z))
                ez_loss = dz_criterion(d_z, torch.ones_like(d_z))
                dz_loss_tot = dz_loss + dz_loss_prior
                loss = eg_loss + 0*reg_loss + 0.0001*ez_loss

                dz_optimizer.zero_grad()
                dz_loss_tot.backward(retain_graph=True)
                dz_optimizer.step()

                eg_optimizer.zero_grad()
                loss.backward(retain_graph=True)
                eg_optimizer.step()
                #dz_loss = z_criterion(d_z_prior, d_z)

                now = datetime.datetime.now()

                epoch_loss += loss.item()
                if save_count % 500 == 0:
                    save_count = 0
                    logging.info('[{h}:{m}[Epoch {e}, i: {c}] Loss: {t}'.format(h=now.hour, m=now.minute, e=epoch, c=i,
                                                                                t=loss.item()))
                    print(f"[{now.hour:d}:{now.minute:d}] [Epoch {epoch:d}, i {i:d}] Loss: {loss.item():f}")
                    cp_path = self.save(name)
                    # joined_image = one_sided(torch.cat((images, generated), 0))
                    # save_image(joined_image, os.path.join(cp_path, 'reconstruct.png'))
                save_count += 1
            save_count = 0
            cp_path = self.save(name)
            with torch.no_grad():  # validation

                self.eval()  # move to eval mode

                z = self.E(validate_images)
                z_l = torch.cat((z, validate_labels), 1)
                generated = self.G(z_l)
                loss = nn.functional.l1_loss(validate_images, generated)
                joined_image = one_sided(generated)
                #torchvision.utils.save_image(generated, 'results/img_' + str(epoch) + '.png', nrow=8)
                torchvision.utils.save_image(joined_image, 'results/onesided_' + str(epoch) + '.png', nrow=8)
                epoch_loss_valid += loss.item()

            loss_tracker.append(epoch_loss / i, epoch_loss_valid / ii, cp_path)
            logging.info('[{h}:{m}[Epoch {e}] Train Loss: {t} Vlidation Loss: {v}'.format(h=now.hour, m=now.minute,
                                                                                              e=epoch, t=loss_tracker.train_losses[-1],
                                                                                              v=loss_tracker.valid_losses[-1]))
        loss_tracker.plot()

    def to(self, device):
        for subnet in self.subnets:
            subnet.to(device=device)

    def cpu(self):
        for subnet in self.subnets:
            subnet.cpu()

    def cuda(self):
        for subnet in self.subnets:
            subnet.cuda()

    def eval(self):
        for subnet in self.subnets:
            subnet.eval()

    def train(self):
        for subnet in self.subnets:
            subnet.train()

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

