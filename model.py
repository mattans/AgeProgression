import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.nn.functional import relu
import itertools
from torch.autograd import Variable

IMAGE_DIMS = torch.Tensor([3] + 2*[128])
MOCK_IMAGE = torch.rand(tuple(IMAGE_DIMS))
IMAGE_LENGTH = IMAGE_DIMS.data[1]
IMAGE_DEPTH = IMAGE_DIMS.data[0]
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
                nn.Conv2d(
                    in_channels=(NUM_ENCODER_CHANNELS * (2**(i-1))) if i > 0 else int(IMAGE_DEPTH),
                    out_channels=NUM_ENCODER_CHANNELS * (2**i),
                    kernel_size=STEP_SIZE,
                    stride=STEP_SIZE
                )
            )
            conv_layers.append(
                nn.ReLU()
            )

        self.convs = nn.Sequential(*conv_layers)
        self.fc = nn.Linear(
            in_features=NUM_ENCODER_CHANNELS * (2**(num_conv_layers-1)),
            out_features=NUM_Z_CHANNELS
        )

    def forward(self, input_face):
        out = input_face
        out = self.convs(out)
        print(1)
        out = out.view(out.size(0), -1) # flatten tensor
        z = self.fc(out)
        print(2)
        return z

class Net(object):
    def __init__(self):
        self.E = Encoder()

    def to(self, device):
        self.E.to(device=device)

    def __call__(self, x):
        return self.E(x)


# Turn image into a batch of size 1, 128x128, RGB
MOCK_IMAGE.unsqueeze_(0)
MOCK_IMAGE = MOCK_IMAGE.to(device)

net = Net()
net.to(device=device)
print(net(MOCK_IMAGE))
print(device)