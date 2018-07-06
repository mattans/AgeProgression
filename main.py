import model
import consts

import re
import numpy as np

import random
import torch







if 'net' not in globals():  # for interactive execution in PyCharm
    net = model.Net()
    net.to(device=consts.device)

    print(consts.device)

    MOCK_TEST = False
    if MOCK_TEST:
        z_mock = net.E(consts.MOCK_IMAGE)
        dz_mock = net.Dz(consts.z_mocl)
        output_mock = net.G(z_mock, consts.MOCK_AGES, consts.MOCK_GENDERS)
        print(output_mock.size())

