import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from methods.er import ER
from utils import *

class MIR(ER):
    def __init__(self, model, buffer, args):
        super().__init__(model, buffer, args)

        self.buffer.sample = self.buffer.sample_mir
        self.sample_kwargs['head_only'] = args.mir_head_only

    def observe(self, inc_data):
        """ full step of processing and learning from data """

        # MIR requires a backward pass before the sampling process
        # we therefore have to switch a few things up

        self.opt.zero_grad()

        # --- training --- #
        inc_loss = self.process_inc(inc_data)
        inc_loss.backward()

        re_data = None
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
                re_loss.backward()

        self.opt.step()

        # --- buffer overhead --- #
        self.buffer.add(inc_data)
