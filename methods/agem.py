import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from methods.er import ER
from utils import *

class AGEM(ER):
    def __init__(self, model, buffer, args):
        super().__init__(model, buffer, args)

        self.grad_dims = []
        for param in model.parameters():
            self.grad_dims += [param.data.numel()]

        device = next(model.parameters()).device
        self.grad_inc = torch.zeros(np.sum(self.grad_dims)).to(device)
        self.grad_re  = torch.zeros(np.sum(self.grad_dims)).to(device)

    def overwrite_grad(self, projected_grad):
        overwrite_grad(self.model.parameters, projected_grad, self.grad_dims)

    def observe(self, inc_data):
        """ full step of processing and learning from data """

        self.opt.zero_grad()

        # --- training --- #
        inc_loss = self.process_inc(inc_data)
        inc_loss.backward()

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
                # store grad
                store_grad(self.model.parameters, self.grad_inc, self.grad_dims)

                # clear grad buffers
                self.model.zero_grad()

                # rehearsal grad
                re_loss = self.process_re(re_data)
                re_loss.backward()
                store_grad(self.model.parameters, self.grad_re, self.grad_dims)

                # potentially project incoming gradient
                dot_p = torch.dot(self.grad_inc, self.grad_re)
                if dot_p < 0.:
                    proj_grad = project(gxy=self.grad_inc, ger=self.grad_re)
                else:
                    proj_grad = self.grad_inc

                self.overwrite_grad(proj_grad)

        self.opt.step()

        # --- buffer overhead --- #
        self.buffer.add(inc_data)


class AGEMpp(AGEM):
    def overwrite_grad(self, projected_grad):
        overwrite_grad(self.model.parameters, projected_grad + self.grad_re, self.grad_dims)


