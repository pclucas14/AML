import torch
import torch.nn as nn
import torch.nn.functional as F
from fvcore.nn import FlopCountAnalysis as FCA

import numpy as np

from methods.er import ER
from utils import *

class MIR(ER):
    def __init__(self, model, logger, train_tf, args):
        super().__init__(model, logger, train_tf, args)

        self.buffer.sample = self.buffer.sample_mir

        self.sample_kwargs.update({
            'lr': args.lr,
            'model': self.model,
            'subsample': args.subsample,
            'head_only': args.mir_head_only,
        })

    @property
    def cost(self):
        bs, bbs = self.args.batch_size, self.args.buffer_batch_size

        # this is the cost of the processing of incoming and rehearsal data
        base_cost = 2 * (bs + bbs) / bs

        # we must also consider the cost of fetching the buffer data
        if self.args.mir_head_only:
            input = torch.FloatTensor(size=(1,) + self.args.input_size).to(self.device)
            total_flops = FCA(self.model, input).total()
            hid   = self.model.return_hidden(input)
            head_flops  = FCA(self.model.linear, hid).total()

            head_rel_cost = (head_flops / total_flops)

            # 1) do the backward pass to get temp model
            # setting to 0 as we could cache this computation and avoid it in `inc_loss.backward()`
            fetch_cost = 0 #2 * head_rel_cost * bs / bs

            # 2) forward the subsampled buffer items in the base model
            fetch_cost += (self.args.subsample - bbs) / bs

            # 3) forward the subsampled buffer items in temp model
            fetch_cost += head_rel_cost * self.args.subsample / bs

        else:
            # 1) do the backward pass to get temp model
            # setting to 0 as we could cache this computation and avoid it in `inc_loss.backward()`
            fetch_cost = 0 #2 * bs / bs

            # 2) forward the subsampled buffer items in the base model
            fetch_cost += (self.args.subsample - bbs) / bs

            # 3) forward the subsampled buffer items in temp model
            fetch_cost += self.args.subsample / bs


        return base_cost + fetch_cost


    def observe(self, inc_data):
        """ full step of processing and learning from data """

        # MIR requires a backward pass before the sampling process
        # we therefore have to switch a few things up

        # keep track of current task for task-based methods
        self.task.fill_(inc_data['t'])

        self.opt.zero_grad()

        # --- training --- #
        inc_loss = self.process_inc(inc_data)
        inc_loss.backward()

        re_data = None
        if len(self.buffer) > 0:

            # -- rehearsal starts ASAP. No task id is used
            if self.args.task_free or self.task > 0:
                re_data = self.buffer.sample(
                        **self.sample_kwargs
                )

                re_loss = self.process_re(re_data)
                re_loss.backward()

        self.opt.step()

        # --- buffer overhead --- #
        self.buffer.add(inc_data)
