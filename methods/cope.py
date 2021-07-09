import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from copy import deepcopy

from methods.er import ER

""" This is still incomplete, need to double check a few things """

class CoPE(ER):
    def __init__(self, model, logger, train_tf, args):
        super().__init__(model, logger, train_tf, args)

        n_classes = model.linear.weight.size(0)
        self.momentum = args.momentum
        self.cope_temperature = args.cope_temperature

        # creating as buffer so no grad is sent back
        self._prototypes  = torch.rand_like(model.linear.weight).uniform_(0, 1)
        self.proto_labels = torch.arange(n_classes).to(self.device)

        delattr(model, 'linear')

        #self.buffer.sample = self.buffer.sample_balanced
        #self.buffer.add    = self.buffer.add_balanced
        self.seen_so_far = torch.LongTensor(size=(0,)).to(self.device)

    @property
    def name(self):
        args = self.args
        return f'CoPE_{args.dataset}_M{args.mem_size}_Augs{args.use_augs}_TF{args.task_free}'

    @property
    def prototypes(self):
        return F.normalize(self._prototypes, dim=-1)[self.seen_so_far]


    def _process(self, data):
        """ get a loss signal from data """

        # potentially augment data
        aug_data = self.train_tf(data['x'])

        hidden   = F.normalize(self.model.return_hidden(aug_data), dim=-1)
        protos   = self.prototypes

        n_inc   = data['x'].size(0)
        n_proto = protos.size(0)
        all_hid    = torch.cat((hidden, protos))
        all_labels = torch.cat((data['y'], self.seen_so_far))

        same_label = all_labels.view(1, -1) == all_labels.view(-1, 1)
        diff_mask  = ~same_label

        # mask out same sample
        pos_mask   = same_label
        pos_mask.fill_diagonal_(False)

        # compute distances across
        similarity = torch.exp(torch.mm(all_hid, all_hid.T) / self.cope_temperature)

        pos_sim = similarity * pos_mask
        neg_sim = similarity * diff_mask

        # similiarity with incoming data
        pos_inc = pos_sim[:n_inc, :n_inc].sum()
        neg_inc = neg_sim[:n_inc, :n_inc].sum()

        # similarity with prototypes
        pos_proto = pos_sim[:n_inc, -n_proto:].sum()
        neg_proto = neg_sim[:n_inc, -n_proto:].sum()

        loss = -(torch.mean(torch.log(pos_sim / (pos_sim + neg_sim))))

        num  = pos_inc + pos_proto
        deno = pos_inc + pos_proto + neg_inc + neg_proto

        new_loss = - (num / deno).log().mean()

        self.n_fwd += data['x'].size(0)
        self.n_bwd += data['x'].size(0)

        return new_loss, hidden.detach(), data['y']


    def process_inc(self, inc_data):
        """ get loss from incoming data """

        loss, hid, y = self._process(inc_data)
        self.inc_hid = hid
        self.inc_y   = y

        return loss


    def process_re(self, re_data):
        """ get loss from rehearsal data """

        loss, hid, y = self._process(re_data)
        self.re_hid  = hid
        self.re_y    = y

        return loss


    @torch.no_grad()
    def _update_prototypes(self):
        f_x, y = self.inc_hid, self.inc_y
        hid_size = f_x.size(-1)

        if hasattr(self, 're_hid'):
            f_x = torch.cat((f_x, self.re_hid))
            y   = torch.cat((y, self.re_y))

        protos = f_x.new_zeros(size=(self.args.n_classes, hid_size))
        count  = y.new_zeros(size=(self.args.n_classes,))

        count.scatter_add_(0, y, torch.ones_like(y))

        out_idx = torch.arange(hid_size, device=y.device).view(1, -1) + y.view(-1, 1) * hid_size
        protos  = protos.view(-1).scatter_add(0, out_idx.view(-1), f_x.view(-1)).view_as(protos)

        protos = protos / count.view(-1, 1)

        # momentum update
        self._prototypes[count > 0] = self.momentum * F.normalize(self._prototypes[count > 0], dim=-1) \
                                    + (1 - self.momentum) * protos[count > 0]


    def observe(self, inc_data):
        """ full step of processing and learning from data """

        present = inc_data['y'].unique()
        self.seen_so_far = torch.cat([self.seen_so_far, present]).unique()

        super().observe(inc_data)

        self._update_prototypes()


    @torch.no_grad()
    def predict(self, x):

        # calculate distance matrix between incoming and _centroids
        hid_x  = F.normalize(self.model.return_hidden(x), dim=-1) # bs x D
        # dist   = (self.prototypes.unsqueeze(0) - hid_x.unsqueeze(1)).pow(2).sum(-1)
        # return -dist
        return torch.mm(hid_x, self.prototypes.T)
