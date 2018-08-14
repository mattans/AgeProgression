import consts
from utils import *
import os
from shutil import copyfile
import numpy as np
from collections import OrderedDict, namedtuple
from torchvision import datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
import datetime
from scipy.misc import imsave as ims
import torchvision
from torchvision.datasets import ImageFolder
import logging
# from torch.nn.functional import relu
from torch.utils.data.sampler import SubsetRandomSampler
import scipy.stats as stats
import cv2


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
        out = self.fc_layer(out)
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

        self.layers.add_module('dz_fc_%d' % (i + 1), nn.Sequential(
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
        in_dims = (3, 16 + consts.NUM_AGES + consts.NUM_GENDERS, 32, 64)
        out_dims = (16, 32, 64, 128)
        self.conv_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(in_dims[:-1], out_dims[1:]), 1):
            self.conv_layers.add_module(
                'dimg_conv_%d' % i,
                nn.Sequential(
                    nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2),
                    nn.BatchNorm2d(out_dim),
                    nn.ReLU()
                )
            )

        self.fc_1 = nn.Sequential(
            nn.Linear(128 * 8 * 8, 1024),
            nn.LeakyReLU()
        )

        self.fc_2 = nn.Sequential(
            nn.Linear(1024, 1),
            # nn.Sigmoid()
        )

    def _compress(self, x):
        return x.view(x.size(0), -1)

    def forward(self, imgs, labels, device):
        out = imgs

        for i, conv_layer in enumerate(self.conv_layers, 1):
            out = conv_layer(out)
            if i == 1:
                labels_tensor = torch.zeros(torch.Size((out.size(0), labels.size(1), out.size(2), out.size(3))),
                                            device=device)
                for img_idx in range(out.size(0)):
                    for label in range(labels.size(1)):
                        labels_tensor[img_idx, label, :, :] = labels[img_idx, label]
                out = torch.cat((out, labels_tensor), consts.NUM_GENDERS/2)

        # run fcs
        out = self._compress(out)
        for fc_layer in self.fc_layers:
            out = fc_layer(out)

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
            label = Label(age, gender).to_tensor() \
                if (isinstance(age, int) and isinstance(gender, int)) \
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
        self.DImg = DiscriminatorImg()
        self.eg_optimizer = Adam(list(self.E.parameters()) + list(self.G.parameters()))
        self.dz_optimizer = Adam(self.Dz.parameters())
        self.im_optimizer = Adam(self.DImg.parameters())
        self.device = None
        if torch.cuda.is_available():
            self.cuda()
            print("On CUDA")
        else:
            self.cpu()
            print("On CPU")

    def __call__(self, *args, **kwargs):
        self.test_single(*args, **kwargs)

    def __repr__(self):
        return os.linesep.join([repr(subnet) for subnet in (self.E, self.Dz, self.G)])

    def test_single(self, img_tensor, age, gender, target):
        # TODO - add the age caption on the original image

        self.eval()
        batch = img_tensor.repeat(consts.NUM_AGES, 1, 1, 1)  # N x D x H x W
        batch.to(self.device)
        z = self.E(batch)  # N x Z
        z.to(self.device)

        gender_tensor = -torch.ones(consts.NUM_GENDERS)
        gender_tensor[int(gender)] *= -1
        gender_tensor = gender_tensor.repeat(consts.NUM_AGES, 1)  # apply gender on all images

        age_tensor = -torch.ones(consts.NUM_AGES, consts.NUM_AGES)
        for i in range(consts.NUM_AGES):
            age_tensor[i][i] *= -1  # apply the i'th age group on the i'th image

        l = torch.cat((age_tensor, gender_tensor), 1)
        l.to(self.device)
        z_l = torch.cat((z, l), 1)

        generated = self.G(z_l)

        img_tensor = img_tensor.transpose(0, 1).transpose(1, 2)
        img_tensor = 255 * one_sided(img_tensor.numpy())
        img_tensor = np.ascontiguousarray(img_tensor, dtype=np.uint8)

        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (2, 25)
        fontScale = 0.5
        fontColor = (0, 128, 0)
        lineType = 2
        cv2.putText(
            img_tensor,
            '{}, {}'.format(["Male", "Female"][gender], age),
            bottomLeftCornerOfText,
            font,
            fontScale,
            fontColor,
            lineType,

        )
        img_tensor = two_sided(torch.from_numpy(img_tensor / 255.0)).float()
        img_tensor = img_tensor.transpose(0, 1).transpose(0, 2)

        joined = torch.cat((img_tensor.unsqueeze(0), generated), 0)

        torchvision.utils.save_image(tensor=joined, filename=os.path.join(target, 'menifa.png'), nrow=joined.size(0), normalize=True, range=(-1, 1))

    def teach(
            self,
            utkface_path,
            batch_size=64,
            epochs=1,
            weight_decay=1e-5,
            lr=2e-4,
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
            validate_images = images.to(device=self.device)
            labels = torch.stack(
                [str_to_tensor(idx_to_class[l]).to(device=self.device) for l in list(labels.numpy())])
            validate_labels = labels.to(device=self.device)

        torchvision.utils.save_image(tensor=validate_images, filename="./results/base.png", normalize=True, range=(-1, 1))

        for optimizer in (self.eg_optimizer, self.dz_optimizer, self.im_optimizer):
            for param in ('weight_decay', 'betas', 'lr'):
                optimizer.param_groups[0][param] = locals()[param]

        #  TODO - write a txt file with all arguments to results folder

        loss_tracker = LossTracker('train', 'valid', 'dz', 'reg')
        save_count = 0
        for epoch in range(1, epochs + 1):
            epoch_eg_loss = []
            epoch_eg_valid_loss = []
            epoch_tv_loss = []
            epoch_uni_loss = []
            for i, (images, labels) in enumerate(train_loader, 1):

                if images.size(0) != batch_size:
                    continue  # tail batch, we can ignore it

                self.train()  # move to train mode

                images = images.to(device=self.device)
                labels = torch.stack([str_to_tensor(idx_to_class[l]).to(device=self.device)
                                      for l in list(labels.numpy())])
                labels = labels.to(device=self.device)
                # print ("DEBUG: iteration: "+str(i)+" images shape: "+str(images.shape))
                z = self.E(images)

                z_l = torch.cat((z, labels), 1)
                generated = self.G(z_l)
                eg_loss = F.l1_loss(generated, images)
                epoch_eg_loss.append(eg_loss.item())

                reg = (
                        torch.sum(torch.abs(generated[:, :, :, :-1] - generated[:, :, :, 1:])) +
                        torch.sum(torch.abs(generated[:, :, :-1, :] - generated[:, :, 1:, :]))
                ) / batch_size
                reg_loss = 0.000 * F.l1_loss(reg, torch.zeros_like(reg))
                reg_loss.to(self.device)
                epoch_tv_loss.append(reg_loss.item())

                ####D_Z####
                z_prior = two_sided(torch.rand_like(z))  # [-1 : 1]
                d_z_prior = self.Dz(z_prior.to(device=self.device))
                d_z = self.Dz(z)

                ###D_IMG###
                d_img_input = self.DImg(images, labels)
                d_img = self.DImg(generated, labels)
                d_img_loss_input = F.binary_cross_entropy_with_logits(d_img_input, torch.ones_like(d_img_input))
                d_img_loss_G = F.binary_cross_entropy_with_logits(d_img, torch.zeros_like(d_img))
                g_img_loss = 0.0001 * F.binary_cross_entropy_with_logits(d_img, torch.ones_like(d_img))
                d_img_loss_tot = 0.1 * (d_img_input + d_img_loss_G)

                dz_loss_prior = F.binary_cross_entropy_with_logits(d_z_prior, torch.ones_like(d_z_prior))
                dz_loss = F.binary_cross_entropy_with_logits(d_z, torch.zeros_like(d_z))
                ez_loss = 0.0001 * F.binary_cross_entropy_with_logits(d_z, torch.ones_like(d_z))
                ez_loss.to(self.device)
                dz_loss_tot = 0.1 * (dz_loss + dz_loss_prior)
                epoch_uni_loss.append(dz_loss_tot.item())

                self.eg_optimizer.zero_grad()
                loss = eg_loss + reg_loss + ez_loss + g_img_loss
                loss.backward(retain_graph=True)
                self.eg_optimizer.step()

                self.dz_optimizer.zero_grad()
                dz_loss_tot.backward()
                self.dz_optimizer.step()

                self.im_optimizer.zero_grad()
                d_img_loss_tot.backward()
                self.im_optimizer.step()

                now = datetime.datetime.now()

                if save_count % 500 == 0:
                    save_count = 0
                    logging.info('[{h}:{m}[Epoch {e}, i: {c}] Loss: {t}'.format(h=now.hour, m=now.minute, e=epoch, c=i,
                                                                                t=loss.item()))
                    print(f"[{now.hour:d}:{now.minute:d}] [Epoch {epoch:d}, i {i:d}] Loss: {loss.item():f}")
                    cp_path = self.save(name)
                save_count += 1
            cp_path = self.save(name) #save at the end

            with torch.no_grad():  # validation

                self.eval()  # move to eval mode

                z = self.E(validate_images)
                z_l = torch.cat((z, validate_labels), 1)
                generated = self.G(z_l)
                loss = F.l1_loss(validate_images, generated)
                joined_image = one_sided(generated)
                torchvision.utils.save_image(tensor=joined_image, filename='results/onesided_' + str(epoch) + '.png', nrow=8, normalize=True, range=(-1, 1))
                epoch_eg_valid_loss.append(loss.item())

            epoch_eg_loss = np.array(epoch_eg_loss)
            epoch_eg_valid_loss = np.array(epoch_eg_valid_loss)
            epoch_tv_loss = np.array(epoch_tv_loss)
            epoch_uni_loss = np.array(epoch_uni_loss)
            print(epoch_eg_loss.mean(), epoch_eg_valid_loss.mean(), epoch_tv_loss.mean(), epoch_uni_loss.mean(), cp_path)
            loss_tracker.append_many(train=epoch_eg_loss.mean(), valid=epoch_eg_valid_loss.mean(), dz=epoch_uni_loss.mean(), reg=epoch_tv_loss.mean())
            loss_tracker.plot()
            logging.info('[{h}:{m}[Epoch {e}] Loss: {l}'.format(h=now.hour, m=now.minute, e=epoch, l=repr(loss_tracker)))
        loss_tracker.plot()

    def _mass_fn(self, fn_name, *args, **kwargs):
        for class_attr in dir(self):
            if not class_attr.startswith('_'):  # ignore private members, for example self.__class__
                class_attr = getattr(self, class_attr)
                if hasattr(class_attr, fn_name):
                    fn = getattr(class_attr, fn_name)
                    fn(*args, **kwargs)

    def to(self, device):
        self._mass_fn('to', device=device)

    def cpu(self):
        self._mass_fn('cpu')
        self.device = torch.device('cpu')

    def cuda(self):
        self._mass_fn('cuda')
        self.device = torch.device('cuda')

    def eval(self):
        self._mass_fn('eval')

    def train(self):
        self._mass_fn('train')

    def save(self, path):
        if not os.path.isdir(path):
            os.mkdir(path)
        path = os.path.join(path, datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
        if not os.path.isdir(path):
            os.mkdir(path)

        for class_attr_name in dir(self):
            if not class_attr_name.startswith('_'):
                class_attr = getattr(self, class_attr_name)
                if hasattr(class_attr, 'state_dict'):
                    state_dict = getattr(class_attr, 'state_dict')
                    torch.save(state_dict, os.path.join(path, "{}.dat".format(class_attr_name)))

        print("Saved to " + path)
        return path

    def load(self, path):

        for class_attr_name in dir(self):
            if not class_attr_name.startswith('_'):
                class_attr = getattr(self, class_attr_name)
                if hasattr(class_attr, 'load_state_dict'):
                    load_state_dict = getattr(class_attr, 'load_state_dict')
                    load_state_dict(torch.load(os.path.join(path, "{}.dat".format(class_attr_name)))())

        print("Loaded from " + path)
