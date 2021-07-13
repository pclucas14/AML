import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from methods.base import Method
from utils import *

class IID(Method):
    def __init__(self, model, logger, train_tf, args):
        super(IID, self).__init__(model, logger, train_tf, args)

        assert args.n_tasks == 1

        args.train_loader.sampler.set_task(0)
        self.loader = iter(args.train_loader)

    @property
    def name(self):
        args = self.args
        return f'IID_{args.dataset[-10:]}_Augs{args.use_augs}'

    @property
    def cost(self):
        return 3 * (self.args.batch_size) / self.args.batch_size

    def _process(self, data):
        """ get a loss signal from data """

        aug_data = self.train_tf(data['x'])
        pred     = self.model(aug_data)
        loss     = self.loss(pred, data['y'])
        return loss


    def observe(self, inc_data):
        """ full step of processing and learning from data """

        for it in range(self.args.n_iters):
            inc_loss = self._process(inc_data)
            self.update(inc_loss)


class IIDpp(IID):
    @property
    def name(self):
        args = self.args
        return f'IID++_{args.dataset[-10:]}_Augs{args.use_augs}'

    @property
    def cost(self):
        return 3 * (self.args.batch_size + self.args.buffer_batch_size) / self.args.batch_size


    def observe(self, inc_data):
        """ full step of processing and learning from data """

        for it in range(self.args.n_iters):
            # --- training --- #
            inc_loss = self._process(inc_data)

            try:
                xx, yy = next(self.loader)
            except StopIteration:
                self.lodaer = iter(self.args.train_loader)
                xx, yy = next(self.loader)

            xx, yy = xx.to(self.device), yy.to(self.device)

            inc_loss += self._process({'x': xx, 'y': yy})

            self.update(inc_loss)
