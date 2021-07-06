import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from methods.base import Method
from utils import *

class ER(Method):
    def __init__(self, model, buffer, args):
        super(ER, self).__init__(model, buffer, args)

        self.sample_kwargs = {
            'lr':        args.lr,
            'amt':       args.buffer_batch_size,
            'model':     self.model,
            'subsample': args.subsample,
        }

    def _process(self, data):
        """ get a loss signal from data """

        pred = self.model(data['x'])
        loss = self.loss(pred, data['y'])
        return loss


    def process_inc(self, inc_data):
        """ get loss from incoming data """

        return self._process(inc_data)


    def process_re(self, re_data):
        """ get loss from rehearsal data """

        return self._process(re_data)


    def observe(self, inc_data):
        """ full step of processing and learning from data """

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


