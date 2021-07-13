import torch
import torch.nn as nn
import torch.nn.functional as F
from fvcore.nn import FlopCountAnalysis as FCA

import numpy as np
from buffer import Buffer


# Abstract Class
class Method():
    def __init__(self, model, logger, train_tf, args):

        self.args     = args
        self.model    = model
        self.train_tf = train_tf
        self.logger   = logger

        self.device = next(model.parameters()).device
        self.buffer = Buffer(capacity=args.mem_size, input_size=args.input_size, device=self.device)

        self.loss = F.cross_entropy
        self.opt  = torch.optim.SGD(self.model.parameters(), lr=args.lr)

        self.n_fwd, self.n_bwd = 0, 0

        self.logger.register_name(self.name)

    @property
    def name(self):
        return ''

    @property
    def cost(self):
        """ return the number of passes (fwd + bwd = 2) through the model for training on one sample """

        raise NotImplementedError


    @property
    def infererence_cost(self):

        return self.cost


    @property
    def one_sample_flop(self):
        if not hasattr(self, '_train_cost'):
            input = torch.FloatTensor(size=(1,) + self.args.input_size).to(self.device)
            flops = FCA(self.model, input)
            self._train_cost = flops.total() / 1e6 # MegaFlops

        return self._train_cost


    def observe(self, inc_data, rehearse=False):
        """ full step of processing and learning from data """

        raise NotImplementedError


    def predict(self, x):
        """ used for test time prediction """

        return self.model(x)


    def update(self, loss):
        """ update parameters from loss """

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()


    def train(self):

        self.model.train()


    def eval(self):

        self.model.eval()
