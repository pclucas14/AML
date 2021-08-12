import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from copy import deepcopy
from methods.base import Method
from utils import *

class SSIL(Method):
    def __init__(self, model, logger, train_tf, args):
        super(SSIL, self).__init__(model, logger, train_tf, args)

        # note that this is not used for task-free methods
        self.task = torch.LongTensor([0]).to(self.device)
        self._task_labels = torch.arange(args.n_classes_per_task).to(self.device)

        self.distill = args.distill_coef > 0
        print(f'distillation {self.distill}')

        self.sample_kwargs = {
            'amt':          args.buffer_batch_size,
            'exclude_task': None if args.task_free else self.task,
        }

        #self.buffer.sample = self.buffer.sample_balanced

    @property
    def name(self):
        args = self.args
        return f'SSIL_DS{args.dataset[-10:]}_M{args.mem_size}_Augs{args.use_augs}_DC{args.distill_coef}'

    @property
    def cost(self):
        return 3 * (self.args.batch_size + self.args.buffer_batch_size) / self.args.batch_size

    @property
    def task_labels(self):
        return self._task_labels + self.task * self.args.n_classes_per_task

    def _on_task_switch(self):
        self._old_model = deepcopy(self.model)
        self._old_model.eval()

    def process_inc(self, data):
        """ get a loss signal from data """

        aug_data = self.train_tf(data['x'])
        pred     = self.model(aug_data)

        # always mask out logits not present in the current batch
        mask = torch.zeros_like(pred)
        mask[:, self.task_labels] = 1
        m_pred = pred.masked_fill(mask == 0, -1e9)

        loss     = self.loss(m_pred, data['y'])

        # task-wise distillation loss
        if self.distill and self.task > 0:
            ub = self.task_labels.min()

            pred = pred.reshape(-1, self.args.n_tasks, self.args.n_classes_per_task)
            p, log_p = F.softmax(pred, -1), F.log_softmax(pred, -1)

            with torch.no_grad():
                tgt_pred  = self._old_model(aug_data)
                tgt_pred = tgt_pred.reshape(-1, self.args.n_tasks, self.args.n_classes_per_task)
                tgt_p, tgt_log_p = F.softmax(tgt_pred, -1), F.log_softmax(tgt_pred, -1)

            # distill all but current task
            out = tgt_p * (tgt_log_p - log_p)
            out = out.mean(0).sum()

            loss += out

        return loss

    def process_re(self, data):
        """ get a loss signal from data """

        aug_data = self.train_tf(data['x'])
        pred     = self.model(aug_data)

        # always mask out logits not present in the current batch
        mask = torch.zeros_like(pred)
        mask[:, :self.task_labels.min()] = 1
        m_pred = pred.masked_fill(mask == 0, -1e9)

        loss     = self.loss(m_pred, data['y'])

        # task-wise distillation loss
        if self.distill and self.task > 0:
            ub = self.task_labels.min()

            pred = pred.reshape(-1, self.args.n_tasks, self.args.n_classes_per_task)
            p, log_p = F.softmax(pred, -1), F.log_softmax(pred, -1)

            with torch.no_grad():
                tgt_pred  = self._old_model(aug_data)
                tgt_pred = tgt_pred.reshape(-1, self.args.n_tasks, self.args.n_classes_per_task)
                tgt_p, tgt_log_p = F.softmax(tgt_pred, -1), F.log_softmax(tgt_pred, -1)

            # distill all but current task
            out = tgt_p * (tgt_log_p - log_p)
            out = out.mean(0).sum()

            loss += out

        return loss


    def observe(self, inc_data):
        """ full step of processing and learning from data """
        if inc_data['t'] != self.task:
            self._on_task_switch()

        # keep track of current task for task-based methods
        self.task.fill_(inc_data['t'])

        for it in range(self.args.n_iters):
            # --- training --- #
            inc_loss = self.process_inc(inc_data)

            re_loss  = 0
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


