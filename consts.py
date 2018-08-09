import torch
import numpy as np
import random
import re
IMAGE_PATH = "./data/UTKFace"
EPOCHS = 100
NUM_OF_MOCK_IMGS = np.random.randint(2, 16)
IMAGE_DIMS = torch.Tensor([NUM_OF_MOCK_IMGS, 3, 128, 128])
# MOCK_IMAGES = torch.rand(tuple(IMAGE_DIMS))
IMAGE_LENGTH = IMAGE_DIMS.data[2]
IMAGE_DEPTH = IMAGE_DIMS.data[1]
BATCH_SIZE = 64
KERNEL_SIZE = 2
STRIDE_SIZE = 2
NUM_ENCODER_CHANNELS = 64
NUM_Z_CHANNELS = 100
NUM_GEN_CHANNELS = 1024
NUM_AGES = 10
# MOCK_AGES = -torch.ones(NUM_OF_MOCK_IMGS, NUM_AGES)
NUM_GENDERS = 2
# MOCK_GENDERS = -torch.ones(NUM_OF_MOCK_IMGS, NUM_GENDERS)
UTKFACE_ORIGINAL_IMAGE_FORMAT = re.compile('^(\d+)_(\d+)_\d+_(\d+)\.jpg\.chip\.jpg$')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# MOCK_IMAGES = MOCK_IMAGES.to(device)
# MOCK_AGES = MOCK_AGES.to(device)
# MOCK_GENDERS = MOCK_GENDERS.to(device)
for i in range(NUM_OF_MOCK_IMGS):
    pass
    # MOCK_GENDERS[i][random.getrandbits(1)] *= -1  # random hot gender
    # MOCK_AGES[i][random.randint(0, NUM_AGES - 1)] *= -1  # random hot age
# MOCK_LABELS = torch.cat((MOCK_AGES, MOCK_GENDERS), 1)
MALE = 0
FEMALE = 1
