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
from torchvision.utils import save_image
from collections import defaultdict
import imageio
import cv2



fmt_t = "%H_%M"
fmt = "%Y_%m_%d"

######################################################################
# Name: save_image_normalized
# Description: Save tensor as am .png image in the file system.
######################################################################
def save_image_normalized(*args, **kwargs):
    save_image(*args, **kwargs, normalize=True, range=(-1, 1))


######################################################################
# Name: merge
# Description:
######################################################################
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


######################################################################
# Name:
# Description:
######################################################################
def two_sided(x):
    return 2 * (x - 0.5)

######################################################################
# Name:
# Description:
######################################################################
def one_sided(x):
    return (x + 1) / 2


######################################################################
# Name:
# Description:
######################################################################
pil_to_model_tensor_transform = transforms.Compose(
    [
        transforms.Resize(size=(128, 128)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 2 * x - 1)  # [0:1] -> [-1:1]
    ]
)

######################################################################
# Name:
# Description:
######################################################################
def get_utkface_dataset(root):
    print(root)
    ret = lambda: ImageFolder(os.path.join(root, 'labeled'), transform=pil_to_model_tensor_transform)
    try:
        return ret()
    except (RuntimeError, FileNotFoundError):
        sort_to_classes(os.path.join(root, 'unlabeled'), print_cycle=1000)
        return ret()

######################################################################
# Name:
# Description:
######################################################################
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


######################################################################
# Name:
# Description:
######################################################################
def get_fgnet_person_loader(root):
    return DataLoader(dataset=ImageFolder(root, transform=pil_to_model_tensor_transform), batch_size=1)


######################################################################
# Name:
# Description:
######################################################################
def str_to_tensor(text):
    age_group, gender = text.split('.')
    age_tensor = -torch.ones(consts.NUM_AGES)
    age_tensor[int(age_group)] *= -1
    gender_tensor = -torch.ones(consts.NUM_GENDERS)
    gender_tensor[int(gender)] *= -1
    result = torch.cat((age_tensor, gender_tensor), 0)
    return result


######################################################################
# Name:
# Description:
######################################################################
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


######################################################################
# Name:
# Description:
######################################################################
def optimizer_and_criterion(criter_class, optim_class, *modules, **optim_args):
    params = []
    for module in modules:
        params.extend(list(module.parameters()))
    optimizier = optim_class(params=params, **optim_args)
    return optimizier, criter_class(reduction='elementwise_mean')



######################################################################
# Name:
# Description:
######################################################################
def default_train_results_dir(eval=True):
    return os.path.join('.', 'trained_models', datetime.datetime.now().strftime(fmt) if eval else fmt)


######################################################################
# Name:
# Description:
######################################################################
def default_where_to_save(eval=True):
    path_str = os.path.join('.', 'results', datetime.datetime.now().strftime(fmt), datetime.datetime.now().strftime(fmt_t))
    if not os.path.exists(path_str):
        os.makedirs(path_str)


######################################################################
# Name:
# Description:
######################################################################
def default_test_results_dir(eval=True):
    return os.path.join('.', 'test_results', datetime.datetime.now().strftime(fmt) if eval else fmt)


######################################################################
# Name:
# Description:
######################################################################
class LossTracker(object):
    def __init__(self, *names, **kwargs):
        assert 'train' in names and 'valid' in names, str(names)
        self.losses = defaultdict(lambda: [])
        self.paths = []
        self.epochs = 0
        self.use_heuristics = kwargs.get('use_heuristics', False)
        self.eps = abs(kwargs.get('eps', 1e-3))
        if(names[-1] == True):
           # print("names[-1] - "+names[-1])
            plt.ion()
            plt.show()
        else:
            plt.switch_backend("agg")


    # deprecated
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

    def append_single(self, name, value):
        self.losses[name].append(value)

    def append_many(self, **names):
        for name, value in names.items():
            self.append_single(name, value)

    def append_many_and_plot(self, **names):
        self.append_many(**names)

    def plot(self):
        print("in plot")
        plt.clf()
        graphs = [plt.plot(loss, label=name)[0] for name, loss in self.losses.items()]
        plt.legend(handles=graphs)
        plt.xlabel('Epochs')
        plt.ylabel('Averaged loss')
        plt.title('Losses by epoch')
        plt.grid(True)
        plt.draw()
        plt.pause(0.001)

    @staticmethod
    def show():
        print("in show")
        plt.show()

    @staticmethod
    def save(path):
        plt.savefig(path, transparent=True)

    def __repr__(self):
        ret = {}
        for name, value in self.losses.items():
            ret[name] = value[-1]
        return str(ret)

######################################################################
# Name:
# Description:
######################################################################
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

######################################################################
# Name:
# Description:
######################################################################
def mean(l):
    return np.array(l).mean()


######################################################################
# Name:
# Description:
######################################################################
from sklearn.metrics.regression import mean_squared_error as mse
def uni_loss(input):
    assert len(input.shape) == 2
    batch_size, input_size = input.size()
    hist = torch.histc(input=input, bins=input_size, min=-1, max=1)
    return mse(hist, batch_size * torch.ones_like(hist)) / input_size


######################################################################
# Name:
# Description:
######################################################################
def create_gif(list_of_img_path , save_dst) :
    frames = []
    index = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (2, 25)
    fontScale = 0.5
    fontColor = (0, 128, 0)  # dark green, should be visible on most skin colors
    lineType = 2

    for path in list_of_img_path:
        image = cv2.imread(path)
        cv2.putText(
            image,
            '{}, {}'.format("Image index: ", index),
            bottomLeftCornerOfText,
            font,
            fontScale,
            fontColor,
            lineType,
        )

        # Save image
        name = os.path.join(save_dst , "img_out_"+str(index)+'.jpeg')
        cv2.imwrite( name , image)
        frame = cv2.imread(name)
        frames.append(frame)
        gif_path = os.path.join(save_dst , 'movie_'+str(index)+'.gif')
        kargs = {'duration': 1}
        imageio.mimsave(gif_path, frames, 'GIF', **kargs)
        index += 1
