import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from methods.er import ER
from utils import *

class DER(ER):
    def __init__(self, model, logger, train_tf, args):
        super(DER, self).__init__(model, logger, train_tf, args)

        assert args.use_augs, 'must be used with augmentations'


    def process_inc(self, inc_data):

        # aug_data = inc_data['x']
        aug_data = self.train_tf(inc_data['x'])
        logits   = self.model(aug_data)
        loss     = self.loss(logits, inc_data['y'])

        self.inc_loss = loss
        # store logits to add to buffer
        inc_data['logit'] = logits.detach()

        return loss


    def process_re(self, re_data):
        """ get loss from rehearsal data """

        # potentially augment data
        aug_data = self.train_tf(re_data['x'])
        logits   = self.model(aug_data)
        loss     =  self.args.alpha * F.mse_loss(logits, re_data['logit'])
        # loss     =  self.args.alpha * F.cosine_similarity(logits, re_data['logit']).mean()

        return loss


class DERpp(DER):
    def __init__(self, model, logger, train_tf, args):
        super().__init__(model, logger, train_tf, args)

        # double the buffer batch size to perform the two ops
        self.sample_kwargs['amt'] = args.buffer_batch_size * 2


    @property
    def cost(self):
        return 2 * (self.args.batch_size + 2 * self.args.buffer_batch_size) / self.args.batch_size


    def process_re(self, re_data):
        """ get loss from rehearsal data """

        # potentially augment data
        aug_data = self.train_tf(re_data['x'])
        output   = self.model(aug_data)
        o1, o2   = output.chunk(2)

        x1, x2 = aug_data.chunk(2)
        y1, y2 = re_data['y'].chunk(2)
        l1, l2 = re_data['logit'].chunk(2)

        aa = F.mse_loss(o1, l1)
        bb = self.loss(o2, y2)

        loss  = self.args.alpha * aa
        loss += self.args.beta  * bb

        # print(f'{self.inc_loss.item():.4f}\t {aa.item():.4f}\t {bb.item():.4f}')

        return loss

