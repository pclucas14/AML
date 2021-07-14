import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import numpy as np

from buffer import Buffer
from methods.er_aml import ER_AML
from model import normalize
from utils import *

class MoCo(ER_AML):
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
        return 3 * (self.args.batch_size + self.args.buffer_batch_size) / self.args.batch_size + self.args.batch_size


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


    def process_inc(self, inc_data):
        """ get loss from incoming data """

        n_fwd = 0

        with torch.no_grad():
            ema_hid   = self.ema_model.return_hidden(inc_data['x'])
            self.queue.add({'x': ema_hid.detach(), 'y': inc_data['y']})

        if inc_data['t'] > 0 or (self.args.task_free and len(self.buffer) > 0):
            # do fancy pos neg
            pos_x, neg_x, pos_y, neg_y, invalid_idx, n_fwd = \
                    self.buffer.sample_pos_neg(
                        inc_data,
                        task_free=self.args.task_free,
                        same_task_neg=True
                    )

            # normalized hidden incoming
            hidden  = self.model.return_hidden(inc_data['x'])
            hidden_norm = normalize(hidden[~invalid_idx])

            all_xs  = torch.cat((pos_x, neg_x))
            all_hid = normalize(self.model.return_hidden(all_xs))
            all_hid = all_hid.reshape(2, pos_x.size(0), -1)
            pos_hid, neg_hid = all_hid[:, ~invalid_idx]

            if (~invalid_idx).any():

                inc_y = inc_data['y'][~invalid_idx]
                pos_y = pos_y[~invalid_idx]
                neg_y = neg_y[~invalid_idx]
                hid_all = torch.cat((pos_hid, neg_hid), dim=0)
                y_all   = torch.cat((pos_y, neg_y), dim=0)

                loss = self.sup_con_loss(
                        labels=y_all,
                        features=hid_all.unsqueeze(1),
                        anch_labels=inc_y.repeat(2),
                        anchor_feature=hidden_norm.repeat(2, 1),
                        temperature=self.args.supcon_temperature,
                )

            else:
                loss = 0.

        else:
            # do regular training at the start
            loss = self.loss(self.model(inc_data['x']), inc_data['y'])

        self.n_fwd_inc += (n_fwd + inc_data['x'].size(0))
        self.n_fwd_inc_cnt += 1

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
