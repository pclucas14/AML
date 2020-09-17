import math
import logging
import numpy as np

import torch
from torch import nn
import torch.utils.data
from torch.nn import functional as F
from copy import deepcopy

class EMA(nn.Module):
    def __init__(self, gamma=0.99, update_freq=1):
        super(EMA, self).__init__()

        self.count = 0
        self.gamma = gamma
        self.update_freq = update_freq

        self.model = None


    def update(self, model):

        if self.model is None:
            self.model = deepcopy(model)
            return

        self.count += 1

        if (self.count % self.update_freq) == 0:
            for name_p, ema_p in self.model.state_dict().items():
                ema_p.data.copy_(self.gamma * ema_p + (1. - self.gamma) * model.state_dict()[name_p])


    @torch.no_grad()
    def forward(self, x):
        return self.model(x)

