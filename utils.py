import consts
import os

from shutil import copyfile
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import datetime

def merge(images, size):
    h, w = images.shape[2], images.shape[3]
    img = np.zeros((3, h * size[0], w * size[1]))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = int(idx / size[1])
        img[0][j * h:j * h + h, i * w:i * w + w] = image[0]
        img[1][j * h:j * h + h, i * w:i * w + w] = image[1]
        img[2][j * h:j * h + h, i * w:i * w + w] = image[2]

    return img


pil_to_model_tensor_transform = transforms.Compose(
    [
        transforms.Resize(size=(128, 128)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 2 * x - 1)  # [0:1] -> [-1:1]
    ]
)


def get_utkface_dataset(root):
    ret = lambda: ImageFolder(os.path.join(root, 'labeled'), transform=pil_to_model_tensor_transform)
    try:
        return ret()
    except (RuntimeError, FileNotFoundError):
        sort_to_classes(os.path.join(root, 'unlabeled'), print_cycle=1000)
        return ret()

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
        matcher = consts.UTKFACE_ORIGINAL_IMAGE_FORMAT.match(f)
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


def get_fgnet_person_loader(root):
    return DataLoader(dataset=ImageFolder(root, transform=pil_to_model_tensor_transform), batch_size=1)


def str_to_tensor(text):
    age_group, gender = text.split('.')
    age_tensor = -torch.ones(consts.NUM_AGES)
    age_tensor[int(age_group)] *= -1
    gender_tensor = -torch.ones(consts.NUM_GENDERS)
    gender_tensor[int(gender)] *= -1
    result = torch.cat((age_tensor, gender_tensor), 0)
    return result


class Label(namedtuple('Label', ('age', 'gender'))):
    def __init__(self, age, gender):
        super(Label, self).__init__()
        _age = self.age - 1
        if _age < 20:
            self.age_group = max(_age // 5, 0)  # first 4 age groups are for kids <= 20, 5 years intervals
        else:
            self.age_group = min(4 + (_age - 20) // 10, consts.NUM_AGES - 1)  # last (6?) age groups are for adults > 20, 10 years intervals

    def to_str(self):
        return '%d.%d' % (self.age_group, self.gender)

    def to_tensor(self):
        return str_to_tensor(self.to_str())


def two_sided(x):
    return 2 * (x - 0.5)


def one_sided(x):
    return (x + 1) / 2


def optimizer_and_criterion(criter_class, optim_class, *modules, **optim_args):
    params = []
    for module in modules:
        params.extend(list(module.parameters()))
    optimizier = optim_class(params=params, **optim_args)
    return optimizier, criter_class(reduction='elementwise_mean')


fmt = "%Y_%m_%d___%H_%M_%S"


def default_train_results_dir(eval=True):
    return os.path.join('.', 'trained_models', datetime.datetime.now().strftime(fmt) if eval else fmt)


def default_test_results_dir(eval=True):
    return os.path.join('.', 'test_results', datetime.datetime.now().strftime(fmt) if eval else fmt)


class LossTracker(object):
    def __init__(self, use_heuristics=False, eps=1e-3):
        self.train_losses = []
        self.valid_losses = []
        self.tv_losses = []
        self.uni_losses = []
        self.paths = []
        self.epochs = 0
        self.use_heuristics = use_heuristics
        self.eps = abs(eps)

    def append(self, train_loss, valid_loss, tv_loss, uni_loss, path):
        self.train_losses.append(train_loss)
        self.valid_losses.append(valid_loss)
        self.tv_losses.append(tv_loss)
        self.uni_losses.append(uni_loss)
        self.paths.append(path)
        self.epochs += 1
        if self.use_heuristics and self.epochs >= 2:
            delta_train = self.train_losses[-1] - self.train_losses[-2]
            delta_valid = self.valid_losses[-1] - self.valid_losses[-2]
            if delta_train < -self.eps and delta_valid < -self.eps:
                pass  # good fit, continue training
            elif delta_train < -self.eps and delta_valid > +self.eps:
                pass  # overfit, consider stop the training now
            elif delta_train > +self.eps and delta_valid > +self.eps:
                pass  # underfit, if this is in an advanced epoch, break
            elif delta_train > +self.eps and delta_valid < -self.eps:
                pass  # unknown fit, check your model, optimizers and loss functions
            elif 0 < delta_train < +self.eps and self.epochs >= 3:
                prev_delta_train = self.train_losses[-2] - self.train_losses[-3]
                if 0 < prev_delta_train < +self.eps:
                    pass  # our training loss is increasing but in less than eps,
                    # this is a drift that needs to be caught, consider lower eps next time
            else:
                pass  # saturation \ small fluctuations

    def plot(self):

        p0, = plt.plot(self.train_losses, label='Training')
        p1, = plt.plot(self.valid_losses, label='Validation')
        p2, = plt.plot(self.tv_losses, label='TotalVar')
        p3, = plt.plot(self.uni_losses, label='Uniformity')
        plt.legend(handles=[p0, p1, p2, p3])
        plt.xlabel('Epochs')
        plt.ylabel('Averaged loss')
        plt.title('Losses by epoch')
        plt.grid(True)
        plt.show()


def get_list_of_labels(lst):
    new_list = []
    for label in lst:
        if 0 <= label <= 5:
            new_list.append(0)
        elif 6 <= label <= 10:
            new_list.append(1)
        elif 11 <= label <= 15:
            new_list.append(2)
        elif 16 <= label <= 20:
            new_list.append(3)
        elif 21 <= label <= 30:
            new_list.append(4)
        elif 31 <= label <= 40:
            new_list.append(5)
        elif 41 <= label <= 50:
            new_list.append(6)
        elif 51 <= label <= 60:
            new_list.append(7)
        elif 61 <= label <= 70:
            new_list.append(89)
        else:
            new_list.append(9)
        return new_list

