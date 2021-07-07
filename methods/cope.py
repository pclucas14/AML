import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from copy import deepcopy

from methods.er import ER

class CoPE(ER):
    def __init__(self, model, buffer, args):
        super().__init__(model, buffer, args)

        n_classes = model.linear.weight.size(0)
        self.momentum = args.momentum
        self.cope_temperature = args.cope_temperature
        device = next(model.parameters()).device

        # creating as buffer so no grad is sent back
        self._prototypes  = torch.rand_like(model.linear.weight).to(device).uniform_(0, 1. / n_classes)
        self.proto_labels = torch.arange(n_classes).to(device)

        delattr(model, 'linear')

        #self.buffer.sample = self.buffer.sample_balanced
        #self.buffer.add    = self.buffer.add_balanced


    @property
    def prototypes(self):
        return F.normalize(self._prototypes, dim=-1)


    def _process(self, data):
        """ get a loss signal from data """

        hidden = F.normalize(self.model.return_hidden(data['x']), dim=-1)
        protos = self.prototypes

        all_hid    = torch.cat((hidden, protos))
        all_labels = torch.cat((data['y'], self.proto_labels))

        same_label = all_labels.view(1, -1) == all_labels.view(-1, 1)
        diff_mask  = ~same_label

        # mask out same sample
        pos_mask   = same_label
        pos_mask.fill_diagonal_(False)

        # compute distances across
        similarity = torch.exp(torch.mm(all_hid, all_hid.T) / self.cope_temperature)

        pos = torch.sum(similarity * pos_mask, 1).clamp_(min=1e-6)
        neg = torch.sum(similarity * diff_mask, 1).clamp_(min=1e-6)
        loss = -(torch.mean(torch.log(pos / (pos + neg))))

        return loss, hidden.detach(), data['y']


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
        self._prototypes[count > 0] = self.momentum * self.prototypes[count > 0] \
                                    + (1 - self.momentum) * protos[count > 0]


    def observe(self, inc_data):
        """ full step of processing and learning from data """

        # --- To be consistent with the original implementation, we concat
        # --- the incoming and rehearsal data

        # --- training --- #
        inc_loss = self.process_inc(inc_data)

        re_loss, re_data = 0., None
        if len(self.buffer) > 0:

            # -- rehearsal starts ASAP. No task id is used
            if self.args.task_free:
                re_data = self.buffer.sample(
                        **self.sample_kwargs
                )

            # -- rehearsal starts after 1st task. Exclude
            # -- current task labels from the draw.
            elif inc_data['t'] > 0:
                re_data = self.buffer.sample(
                        exclude_task=inc_data['t'],
                        **self.sample_kwargs
                )

            if re_data is not None:
                re_loss = self.process_re(re_data)

        self.update(inc_loss + re_loss)

        # --- buffer overhead --- #
        self.buffer.add(inc_data)



        self._update_prototypes()


    @torch.no_grad()
    def predict(self, x):

        # calculate distance matrix between incoming and _centroids
        hid_x  = F.normalize(self.model.return_hidden(x), dim=-1) # bs x D
        dist   = (self.prototypes.unsqueeze(0) - hid_x.unsqueeze(1)).pow(2).sum(-1)

        return -dist
