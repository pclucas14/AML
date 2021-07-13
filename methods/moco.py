import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity

import copy
import numpy as np

from buffer import Buffer
from methods.er import ER
from utils import *

class MoCo(ER):
    def __init__(self, model, logger, train_tf, args):
        super(MoCo, self).__init__(model, logger, train_tf, args)
        self.M = args.momentum
        self.ema_model = copy.deepcopy(model)

        input = torch.FloatTensor(size=(1,) + args.input_size).to(self.device)
        hid   = self.model.return_hidden(input).shape[1:]

        self.queue = Buffer(capacity=100, input_size=hid, device=self.device)
        self.queue.add = self.queue.add_queue

    @property
    def name(self):
        args = self.args
        return f'ER_DS{args.dataset[-10:]}_M{args.mem_size}_Augs{args.use_augs}_TF{args.task_free}'

    @property
    def cost(self):
        return 3 * (self.args.batch_size + self.args.buffer_batch_size) / self.args.batch_size


    def ema_update(self):
        for ema_p, p in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_p.data = ema_p.data * self.M + p.data * (1. - self.M)


    def _process(self, data):
        """ get a loss signal from data """

        aug_data1 = self.train_tf(data['x'])
        aug_data2 = self.train_tf(data['x'])

        with torch.no_grad():
            ema_hid   = self.ema_model.return_hidden(aug_data1)

        model_hid = self.model.return_hidden(aug_data2)

        self.queue.add({'x': ema_hid.detach(), 'y': data['y']})

        pred     = self.model.linear(model_hid)
        loss     = self.loss(pred, data['y'])

        return loss


    def observe(self, inc_data):
        super().observe(inc_data)

        self.ema_update()


    def predict(self, x):
        # hid = F.normalize(self.model.return_hidden(x), dim=-1)
        hid = F.normalize(self.ema_model.return_hidden(x), dim=-1)
        buf = F.normalize(self.queue.bx, dim=-1)

        # find closest in buffer
        dist = (buf.unsqueeze(0) - hid.unsqueeze(1)).pow(2).sum(-1)

        pred = self.queue.by[dist.argmin(1)]

        return F.one_hot(pred, num_classes=self.args.n_classes)
