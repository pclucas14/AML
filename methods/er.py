import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from methods.base import Method

# Abstract Class
class ER(Method):
    def __init__(self, model, buffer, args):
        super(ER, self).__init__(model, buffer, args)


    def observe(self, inc_data):
        """ full step of processing and learning from data """

        # --- training --- #
        inc_loss = self.process_inc(inc_data)

        re_loss, re_data = 0., None
        if len(self.buffer) > 0:

            # -- rehearsal starts ASAP. No task id is used
            if self.args.task_free:
                re_data = self.buffer.sample(
                        self.args.buffer_batch_size,
                )

            # -- rehearsal starts after 1st task. Exclude
            # -- current task labels from the draw.
            elif inc_data['t'] > 0:
                re_data = self.buffer.sample(
                        self.args.buffer_batch_size,
                        exclude_task=inc_data['t']
                )

            if re_data is not None:
                re_loss = self.process_re(re_data)

        self.update(inc_loss + re_loss)

        # --- buffer overhead --- #
        self.buffer.add(inc_data)


