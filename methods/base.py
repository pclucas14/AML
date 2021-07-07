import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


# Abstract Class
class Method():
    def __init__(self, model, buffer, args):

        self.args   = args
        self.model  = model
        self.buffer = buffer

        self.loss = F.cross_entropy
        self.opt  = torch.optim.SGD(self.model.parameters(), lr=args.lr)


    def update(self, loss):
        """ update parameters from loss """

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()


    def observe(self, inc_data, rehearse=False):
        """ full step of processing and learning from data """

        raise NotImplementedError


    def predict(self, x):
        """ used for test time prediction """

        return self.model(x)


    def train(self):
        self.model.train()


    def eval(self):
        self.model.eval()
