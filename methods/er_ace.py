import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from methods.base import Method
from methods.er   import ER

class ER_ACE(ER):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.seen_so_far = torch.LongTensor(size=(0,)).to(self.buffer.device)

        # only works in the task-free setting
        if not self.args.task_free:
            print('setting ER-ACE as task-free')
            self.args.task_free = True


    def process_inc(self, inc_data):
        """ get loss from incoming data """

        present = inc_data['y'].unique()
        self.seen_so_far = torch.cat([self.seen_so_far, present]).unique()

        # process data
        logits = self.model(inc_data['x'])
        mask   = torch.zeros_like(logits)

        # unmask current classes
        mask[:, present] = 1

        # unmask unseen classes
        mask[:, self.seen_so_far.max():] = 1

        if inc_data['t'] > 0:
            logits  = logits.masked_fill(mask == 0, -1e9)

        loss = self.loss(logits, inc_data['y'])

        return loss

