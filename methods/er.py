import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from methods.base import Method
from utils import *

class ER(Method):
    def __init__(self, model, logger, train_tf, args):
        super(ER, self).__init__(model, logger, train_tf, args)

        # note that this is not used for task-free methods
        self.task = torch.LongTensor([0]).to(self.device)

        self.sample_kwargs = {
            'amt':          args.buffer_batch_size,
            'exclude_task': None if args.task_free else self.task,
        }


    @property
    def cost(self):
        return 2 * (self.args.batch_size + self.args.buffer_batch_size) / self.args.batch_size


    def _process(self, data):
        """ get a loss signal from data """

        aug_data = self.train_tf(data['x'])
        pred     = self.model(aug_data)
        loss     = self.loss(pred, data['y'])
        return loss


    def process_inc(self, inc_data):
        """ get loss from incoming data """

        return self._process(inc_data)


    def process_re(self, re_data):
        """ get loss from rehearsal data """

        return self._process(re_data)


    def observe(self, inc_data):
        """ full step of processing and learning from data """

        # keep track of current task for task-based methods
        self.task.fill_(inc_data['t'])

        for it in range(self.args.n_iters):
            # --- training --- #
            inc_loss = self.process_inc(inc_data)
            assert inc_data['x'].size(0) == inc_data['y'].size(0), pdb.set_trace()

            re_loss, re_data = 0., None
            if len(self.buffer) > 0:

                # -- rehearsal starts ASAP. No task id is used
                if self.args.task_free or self.task > 0:
                    re_data = self.buffer.sample(
                            **self.sample_kwargs
                    )

                    re_loss = self.process_re(re_data)

            self.update(inc_loss + re_loss)

        # --- buffer overhead --- #
        self.buffer.add(inc_data)


