import os
import sys
import pdb
import torch
import torch.nn as nn

import kornia
import numpy as np

from copy import deepcopy
from torchvision import datasets, transforms


class CIFAR:

    default_size = 32
    default_n_tasks = 5

    def base_transforms(H=None):

        if H is None:
            H = CIFAR.default_size

        tfs = transforms.Compose([
            transforms.Resize(H),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2615))
            #lambda x : (x - .5) * 2.
        ])

        return tfs

    def train_transforms(H=None, use_augs=False):

        if H is None:
            H = CIFAR.default_size

        if use_augs:
            tfs = nn.Sequential(
                kornia.augmentation.RandomCrop(size=(H, H), padding=4),
                kornia.augmentation.RandomHorizontalFlip(),
            )
        else:
            tfs = nn.Identity()


        return tfs

    def eval_transforms(H=None):
        return CIFAR.base_transforms(H=H)


class CIFAR10(CIFAR, datasets.CIFAR10):
    pass

class CIFAR100(CIFAR, datasets.CIFAR100):
    pass
