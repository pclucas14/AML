import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from methods.er import ER
from utils import *

class DER(ER):
    def __init__(self, model, train_tf, args):
        super(DER, self).__init__(model, train_tf, args)

        assert args.use_augs, 'must be used with augmentations'

        # we must add an extra buffer to ... the buffer
        self.buffer.add_buffer('logit', torch.FloatTensor, (args.n_classes,))


    def process_re(self, re_data):
        """ get loss from rehearsal data """

        # potentially augment data
        aug_data = self.train_tf(re_data['x'])
        logits   = self.model(aug_data)
        return self.args.alpha * F.mse_loss(logits, re_data['logit'])


class DERpp(DER):
    def __init__(self, model, train_tf, args):
        super().__init__(model, train_tf, args)

        # double the buffer batch size to perform the two ops
        self.sample_kwargs['amt'] = args.buffer_batch_size * 2


    def process_re(self, re_data):
        """ get loss from rehearsal data """

        # potentially augment data
        aug_data = self.train_tf(re_data['x'])
        output   = self.model(aug_data)
        o1, o2   = output.chunk(2)

        x1, x2 = aug_data.chunk(2)
        y1, y2 = re_data['y'].chunk(2)
        l1, l2 = re_data['logit'].chunk(2)

        loss  = self.args.alpha * F.mse_loss(o1, l1)
        loss += self.args.beta  * self.loss(o2, y2)

        return loss

