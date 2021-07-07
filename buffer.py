import numpy as np
import math
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia

from collections import OrderedDict
from collections import Iterable
from utils import *

class Buffer(nn.Module):
    def __init__(
        self,
        capacity,
        items_to_store={'x': (torch.FloatTensor, (3, 32, 32)),
                        'y': (torch.LongTensor,  ()),
                        't': (torch.LongTensor,  ())}
    ):
        super().__init__()

        # we need at least these two attributes
        assert all(key in items_to_store.keys() for key in ['x', 'y'])

        # create placeholders for each item
        self.buffers = []

        for name, (dtype, size) in items_to_store.items():
            tmp = dtype(size=(capacity,) + size).fill_(0)
            self.register_buffer(f'b{name}', tmp)
            self.buffers += [f'b{name}']

        self.cap = capacity
        self.current_index = 0
        self.n_seen_so_far = 0
        self.is_full       = 0

        # defaults
        self.add = self.add_reservoir
        self.sample = self.sample_random

    @property
    def device(self):
        return getattr(self, self.buffers[0]).device


    def __len__(self):
        return self.current_index


    def add_reservoir(self, batch):
        n_elem = batch['x'].size(0)

        place_left = max(0, self.cap - self.current_index)

        indices = torch.FloatTensor(n_elem).to(self.device)
        indices = indices.uniform_(0, self.n_seen_so_far).long()

        if place_left > 0:
            upper_bound = min(place_left, n_elem)
            indices[:upper_bound] = torch.arange(upper_bound) + self.current_index

        valid_indices = (indices < self.cap).long()
        idx_new_data  = valid_indices.nonzero().squeeze(-1)
        idx_buffer    = indices[idx_new_data]

        self.n_seen_so_far += n_elem
        self.current_index = min(self.n_seen_so_far, self.cap)

        if idx_buffer.numel() == 0:
            return

        # perform overwrite op
        for name, data in batch.items():
            buffer = getattr(self, f'b{name}')

            if isinstance(data, Iterable):
                buffer[idx_buffer] = data[idx_new_data]
            else:
                buffer[idx_buffer] = data


    def add_balanced(self, batch):
        n_elem = batch['x'].size(0)

        # increment first
        self.n_seen_so_far += n_elem
        self.current_index = min(self.n_seen_so_far, self.cap)

        # first thing is we just add all the data
        for name, data in batch.items():
            buffer = getattr(self, f'b{name}')

            if not isinstance(data, Iterable):
                data = buffer.new(size=(n_elem, *buffer.shape[1:])).fill_(data)

            buffer = torch.cat((data, buffer))[:self.n_seen_so_far]
            setattr(self, f'b{name}', buffer)

        n_samples_over = buffer.size(0) - self.cap

        # no samples to remove
        if n_samples_over <= 0:
            return

        # remove samples from the most common classes
        class_count   = self.by.bincount()
        rem_per_class = torch.zeros_like(class_count)

        while rem_per_class.sum() < n_samples_over:
            max_idx = class_count.argmax()
            rem_per_class[max_idx] += 1
            class_count[max_idx]   -= 1

        # always remove the oldest samples for each class
        classes_trimmed = rem_per_class.nonzero().flatten()
        idx_remove = []

        for cls in classes_trimmed:
            cls_idx = (self.by == cls).nonzero().view(-1)
            idx_remove += [cls_idx[-rem_per_class[cls]:]]

        idx_remove = torch.cat(idx_remove)
        idx_mask   = torch.BoolTensor(buffer.size(0)).to(self.device)
        idx_mask.fill_(0)
        idx_mask[idx_remove] = 1

        # perform overwrite op
        for name, data in batch.items():
            buffer = getattr(self, f'b{name}')
            buffer = buffer[~idx_mask]
            setattr(self, f'b{name}', buffer)


    def sample_random(self, amt, exclude_task=None, **kwargs):
        buffers = OrderedDict()

        if exclude_task is not None:
            assert hasattr(self, 'bt')
            valid_indices = (self.bt != exclude_task).nonzero().squeeze()
            for buffer_name in self.buffers:
                buffers[buffer_name[1:]] = getattr(self, buffer_name)[valid_indices]
        else:
            for buffer_name in self.buffers:
                buffers[buffer_name[1:]] = getattr(self, buffer_name)[:self.current_index]

        if self.current_index < amt:
            return buffers
        else:
            idx_np = np.random.choice(buffers['x'].size(0), amt, replace=False)
            indices = torch.from_numpy(idx_np).to(self.bx.device)
            by_np = self.by.clone().cpu().data.numpy()
            bx_np = self.bx.clone().cpu().data.numpy()

            return OrderedDict({k:v[indices] for (k,v) in buffers.items()})


    def sample_balanced(self, amt, exclude_task=None, **kwargs):
        buffers = OrderedDict()

        if exclude_task is not None:
            assert hasattr(self, 'bt')
            valid_indices = (self.bt != exclude_task).nonzero().squeeze()
            for buffer_name in self.buffers:
                buffers[buffer_name[1:]] = getattr(self, buffer_name)[valid_indices]
        else:
            for buffer_name in self.buffers:
                buffers[buffer_name[1:]] = getattr(self, buffer_name)[:self.current_index]

        class_count = buffers['y'].bincount()

        # a sample's prob. of being sample is inv. prop to its class abundance
        class_sample_p = 1. / class_count.float() / class_count.size(0)
        per_sample_p   = class_sample_p.gather(0, buffers['y'])
        indices        = torch.multinomial(per_sample_p, amt)

        return OrderedDict({k:v[indices] for (k,v) in buffers.items()})


    def sample_mir(self, amt, subsample, model, exclude_task=None, lr=0.1, head_only=False, **kwargs):
        subsample = self.sample_random(subsample, exclude_task=exclude_task)

        if not hasattr(model, 'grad_dims'):
            model.mir_grad_dims = []
            if head_only:
                for param in model.linear.parameters():
                    model.mir_grad_dims += [param.data.numel()]
            else:
                for param in model.parameters():
                    model.mir_grad_dims += [param.data.numel()]

        if head_only:
            grad_vector = get_grad_vector(list(model.linear.parameters()), model.mir_grad_dims)
            model_temp  = get_future_step_parameters(model.linear, grad_vector, model.mir_grad_dims, lr=lr)
        else:
            grad_vector = get_grad_vector(list(model.parameters()), model.mir_grad_dims)
            model_temp  = get_future_step_parameters(model, grad_vector, model.mir_grad_dims, lr=lr)

        with torch.no_grad():
            hidden_pre  = model.return_hidden(subsample['x'])
            logits_pre  = model.linear(hidden_pre)

            if head_only:
                logits_post = model_temp(hidden_pre)
            else:
                logits_post = model_temp(subsample['x'])

            pre_loss  = F.cross_entropy(logits_pre,  subsample['y'], reduction='none')
            post_loss = F.cross_entropy(logits_post, subsample['y'], reduction='none')

            scores  = post_loss - pre_loss
            indices = scores.sort(descending=True)[1][:amt]

        return OrderedDict({k:v[indices] for (k,v) in subsample.items()})


    def sample_pos_neg(self, inc_data, task_free=True):

        x     = inc_data['x']
        label = inc_data['y']
        task  = torch.zeros_like(label).fill_(inc_data['t'])

        # we need to create an "augmented" buffer containing the incoming data
        bx   = torch.cat((self.bx[:self.current_index], x))
        by   = torch.cat((self.by[:self.current_index], label))
        bt   = torch.cat((self.bt[:self.current_index], task))
        bidx = torch.arange(bx.size(0)).to(bx.device)

        # buf_size x label_size
        same_label = label.view(1, -1)             == by.view(-1, 1)
        same_task  = task.view(1, -1)              == bt.view(-1, 1)
        same_ex    = bidx[-x.size(0):].view(1, -1) == bidx.view(-1, 1)

        task_labels = label.unique()
        real_same_task = same_task

        # TASK FREE METHOD : instead of using the task ID, we'll use labels in
        # the current batch to mimic task
        if task_free:
            same_task = torch.zeros_like(same_task)

            for label_ in task_labels:
                label_exp = label_.view(1, -1).expand_as(same_task)
                same_task = same_task | (label_exp == by.view(-1, 1))

        valid_pos  = same_label & ~same_ex
        valid_neg_same_t = ~same_label & same_task & ~same_ex
        valid_neg_diff_t = ~same_label & ~same_task & ~same_ex

        # remove points which don't have pos, neg from same and diff t
        has_valid_pos = valid_pos.sum(0) > 0
        has_valid_neg = (valid_neg_same_t.sum(0) > 0) & (valid_neg_diff_t.sum(0) > 0)

        invalid_idx = (~has_valid_pos | ~has_valid_neg).nonzero().squeeze()

        if invalid_idx.numel() > 0:
            # so the fetching operation won't fail
            valid_pos[:, invalid_idx] = 1
            valid_neg_same_t[:, invalid_idx] = 1
            valid_neg_diff_t[:, invalid_idx] = 1

        # easier if invalid_idx is a binary tensor
        is_invalid = torch.zeros_like(label).bool()
        is_invalid[invalid_idx] = 1

        # fetch positive samples
        pos_idx = torch.multinomial(valid_pos.float().T, 1).squeeze(1)
        neg_idx_same_t = torch.multinomial(valid_neg_same_t.float().T, 1).squeeze(1)
        neg_idx_diff_t = torch.multinomial(valid_neg_diff_t.float().T, 1).squeeze(1)

        return bx[pos_idx], \
               bx[neg_idx_same_t], \
               bx[neg_idx_diff_t], \
               is_invalid, \
               by[pos_idx], \
               by[neg_idx_same_t], \
               by[neg_idx_diff_t]


