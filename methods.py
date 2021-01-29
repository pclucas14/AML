import argparse
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from copy import deepcopy
from data   import *
from utils  import get_logger, get_temp_logger, logging_per_task, sho_
from buffer import Buffer
from copy   import deepcopy
from pydoc  import locate
from model  import ResNet18, normalize
from utils import naive_cross_entropy_loss, onehot, Lookahead, AverageMeter
import copy


# Abstract Class
class Method():
    def __init__(self, model, buffer, args):
        self.args   = args
        self.model  = model
        self.buffer = buffer

    def observe(self, inc_x, inc_y, inc_t, inc_idx, rehearse=False):
        pass

    def predict(self, x):
        return self.model(x)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()


class ER(Method):
    def __init__(self, model, buffer, args):
        super(ER, self).__init__(model, buffer, args)

    def observe(self, inc_x, inc_y, inc_t, inc_idx, rehearse=False):
        args = self.args

        loss    = F.cross_entropy(self.model(inc_x), inc_y)
        loss_re = 0.
        present = inc_y.unique()

        if rehearse:
            # sample from buffer
            mem_x, mem_y, bt = self.buffer.sample(
                    args.buffer_batch_size,
                    aug=args.use_augmentations,
                    exclude_task=None if args.task_free else inc_t,
                    exclude_labels=present if args.task_free else None
            )

            loss_re = F.cross_entropy(self.model(mem_x), mem_y)

        return loss, loss_re


class ER_ACE(ER):
    def __init__(self, model, buffer, args):
        super(ER_ACE, self).__init__(model, buffer, args)

        self.seen_so_far = torch.LongTensor(size=(0,)).to(buffer.bx.device)

    def observe(self, inc_x, inc_y, inc_t, inc_idx, rehearse=False):

        args = self.args

        present = inc_y.unique()
        self.seen_so_far = torch.cat([self.seen_so_far, present]).unique()

        logits = self.model(inc_x)

        mask = torch.zeros_like(logits)
        mask[:, present] = 1

        if self.seen_so_far.max() < (args.n_classes - 1):
            mask[:, self.seen_so_far.max():] = 1

        if inc_t > 0:
            logits  = logits.masked_fill(mask == 0, -1e9)

        loss    = F.cross_entropy(logits, inc_y)
        loss_re = 0.

        if rehearse:
            # sample from buffer
            mem_x, mem_y, bt = self.buffer.sample(
                    args.buffer_batch_size,
                    aug=args.use_augmentations,
                    exclude_task=None,
                    exclude_labels=None
            )

            loss_re = F.cross_entropy(self.model(mem_x), mem_y)

        return loss, loss_re



class ER_AML(Method):

    def observe(self, inc_x, inc_y, inc_t, inc_idx, rehearse=False):
        args = self.args

        if inc_t == 0:
            loss = F.cross_entropy(self.model(inc_x), inc_y)
        else:
            hidden  = self.model.return_hidden(inc_x)

            # fetch contrastive pairs
            pos_x, neg_x_same_t, neg_x_diff_t, invalid_idx = \
                    self.buffer.fetch_pos_neg_samples(
                            inc_y,
                            inc_t,
                            inc_idx,
                            data=inc_x,
                            task_free=args.task_free)

            if args.buffer_neg > 0:
                all_xs  = torch.cat((pos_x, neg_x_same_t, neg_x_diff_t))
                all_hid = normalize(self.model.return_hidden(all_xs))
                all_hid = all_hid.reshape(3, pos_x.size(0), -1)
                pos_hid, neg_hid_same_t, neg_hid_diff_t = all_hid[:, ~invalid_idx]
            else:
                all_xs  = torch.cat((pos_x, neg_x_same_t))
                all_hid = normalize(self.model.return_hidden(all_xs))
                all_hid = all_hid.reshape(2, pos_x.size(0), -1)
                pos_hid, neg_hid_same_t= all_hid[:, ~invalid_idx]

            hidden_norm = normalize(hidden[~invalid_idx])

            if (~invalid_idx).any():
                loss = args.incoming_neg * \
                        F.triplet_margin_loss(
                                hidden_norm,
                                pos_hid,
                                neg_hid_same_t,
                                args.margin
                        )
                if args.buffer_neg > 0:
                    loss += args.buffer_neg * \
                            F.triplet_margin_loss(
                                    hidden_norm,
                                    pos_hid,
                                    neg_hid_diff_t,
                                    args.margin
                            )
            else:
                loss = 0.

        loss_re = 0.
        if rehearse:
            # sample from buffer
            mem_x, mem_y, bt = self.buffer.sample(
                    args.buffer_batch_size,
                    aug=args.use_augmentations,
                    exclude_task=None,
                    exclude_labels=None
            )

            loss_re = F.cross_entropy(self.model(mem_x), mem_y)

        return loss, loss_re



class AGEM(Method):
    def __init__(self, model, buffer, args):
        super(ER, self).__init__(model, buffer, args)

    def _fetch_grad(self):
        pass

    def observe(self, inc_x, inc_y, inc_t, inc_idx, rehearse=False):
        pass

    def predict(self, x):
        return self.model(x)


class ICARL(Method):
    def __init__(self, model, buffer, args):
        super(ICARL, self).__init__(model, buffer, args)

        assert not args.task_free or args.distill_coef == 0.

        self.D_C = args.distill_coef
        self.distill = args.distill_coef > 0

        self.task = 0
        self._centroids = None
        self._old_model = None

        self.bce_sum = nn.BCEWithLogitsLoss(reduction='sum')

    def _on_task_switch(self):
        self.task += 1

        if self.distill:
            self._old_model = deepcopy(self.model)
            self._old_model.eval()

    @torch.no_grad()
    def _calculate_centroids(self):
        buffer = self.buffer

        n_batches = buffer.x.size(0) // 512 + 1

        hid_size = self.model.return_hidden(buffer.bx[:2]).size(-1)

        arr_D = torch.arange(hid_size).to(buffer.bx.device)

        protos = buffer.bx.new_zeros(size=(self.args.n_classes, hid_size))
        count  = buffer.by.new_zeros(size=(self.args.n_classes,))

        for i in range(n_batches):
            idx    = range(i * 512, min(buffer.x.size(0), (i+1) * 512))
            xx, yy = buffer.bx[idx], buffer.by[idx]

            hid_x = self.model.return_hidden(xx)

            b_proto = torch.zeros_like(protos)
            b_count = torch.zeros_like(count)

            b_count.scatter_add_(0, yy, torch.ones_like(yy))

            out_idx = arr_D.view(1, -1) + yy.view(-1, 1) * hid_size
            b_proto = b_proto.view(-1).scatter_add(0, out_idx.view(-1), hid_x.view(-1)).view_as(b_proto)

            protos += b_proto
            count  += b_count

        self._centroids = protos / count.view(-1, 1)
        self._centroids = self._centroids[count > 0]

    def observe(self, inc_x, inc_y, inc_t, inc_idx, rehearse=False):
        if inc_t != self.task:
            self._on_task_switch()

        args = self.args

        loss   = F.cross_entropy(self.model(inc_x), inc_y)
        logits = self.model(inc_x)

        # build label
        label = torch.zeros_like(logits)
        label[torch.arange(inc_x.size(0)), inc_y] = 1

        loss = self.bce_sum(logits.view(-1), label.view(-1).float()).sum()
        loss = loss / inc_x.size(0)

        # distillation loss
        if self.distill and self._old_model is not None:
            with torch.no_grad():
                tgt = F.sigmoid(self._old_model(inc_x))

            loss += self.D_C * self.bce_sum(logits.view(-1), tgt.view(-1))

        loss_re = 0.
        present = inc_y.unique()

        if rehearse:
            # sample from buffer
            mem_x, mem_y, bt = self.buffer.sample(
                    args.buffer_batch_size,
                    aug=args.use_augmentations,
                    exclude_task=None if args.task_free else inc_t,
                    exclude_labels=present if args.task_free else None
            )

            loss_re = F.cross_entropy(self.model(mem_x), mem_y)
            re_logits = self.model(inc_x)

            # build label
            re_label = torch.zeros_like(re_logits)
            re_label[torch.arange(inc_x.size(0)), mem_y] = 1

            loss_re = self.bce_sum(re_logits.view(-1), re_label.view(-1).float())
            loss_re = loss_re / mem_x.size(0)

        # model updated, centroids no longer valid
        self._centroids = None

        return loss, loss_re


    def predict(self, x):
        if self._centroids is None:
            self._calculate_centroids()

        # calculate distance matrix between incoming and _centroids
        hid_x  = self.model.return_hidden(x) # bs x D
        protos = self._centroids

        dist = (protos.unsqueeze(0) - hid_x.unsqueeze(1)).pow(2).sum(-1)

        return -dist

class ICARL_ACE(ICARL, ER_ACE):
    def __init__(self, model, buffer, args):
        ICARL.__init__(self, model, buffer, args)
        ER_ACE.__init__(self, model, buffer, args)

    def observe(self, *args, **kwargs):
        self._centroids = None

        return ER_ACE.observe(self, *args, **kwargs)

    def predict(self, *args, **kwargs):
        return ICARL.predict(self, *args, **kwargs)

def get_method(method):
    return {'er': ER, 'triplet': ER_AML, 'mask': ER_ACE, 'icarl': ICARL, 'icarl_mask': ICARL_ACE}[method]
