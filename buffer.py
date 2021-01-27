import numpy as np
import math
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
#import torchvision.transforms as transforms
import kornia

class Buffer(nn.Module):
    def __init__(self, args, input_size=None):
        super().__init__()
        self.args = args

        self.place_left = True

        if input_size is None:
            input_size = args.input_size

        buffer_size = args.mem_size
        print('buffer has %d slots' % buffer_size)

        bx = torch.FloatTensor(buffer_size, *input_size).fill_(0)
        by = torch.LongTensor(buffer_size).fill_(0)
        bt = torch.LongTensor(buffer_size).fill_(0)
        bidx = torch.LongTensor(buffer_size).fill_(0)
        logits = torch.FloatTensor(buffer_size, args.n_classes).fill_(0)

        self.current_index = 0
        self.n_seen_so_far = 0
        self.input_size    = input_size

        # registering as buffer allows us to save the object using `torch.save`
        self.register_buffer('bx', bx)
        self.register_buffer('by', by)
        self.register_buffer('bt', bt)
        self.register_buffer('logits', logits)
        self.register_buffer('bidx', bidx)

        self.to_one_hot  = lambda x : x.new(x.size(0), args.n_classes).fill_(0).scatter_(1, x.unsqueeze(1), 1)
        self.arange_like = lambda x : torch.arange(x.size(0)).to(x.device)
        self.shuffle     = lambda x : x[torch.randperm(x.size(0))]

    @property
    def x(self):
        return self.bx[:self.current_index]

    @property
    def y(self):
        return self.to_one_hot(self.by[:self.current_index])

    @property
    def t(self):
        return self.bt[:self.current_index]

    @property
    def valid(self):
        return self.is_valid[:self.current_index]

    def add_reservoir(self, x, y, logits, t, idx, overwrite=True):
        n_elem = x.size(0)
        save_logits = logits is not None

        self.keep = None
        # add whatever still fits in the buffer
        place_left = max(0, self.bx.size(0) - self.current_index)
        if place_left:
            offset = min(place_left, n_elem)
            self.bx[self.current_index: self.current_index + offset].data.copy_(x[:offset])
            self.by[self.current_index: self.current_index + offset].data.copy_(y[:offset])
            self.bt[self.current_index: self.current_index + offset].fill_(t)

            if save_logits:
                self.logits[self.current_index: self.current_index + offset].data.copy_(logits[:offset])

            if idx is not None:
                self.bidx[self.current_index: self.current_index + offset].data.copy_(idx[:offset])

            self.current_index += offset
            self.n_seen_so_far += offset

            # everything was added
            if offset == x.size(0):
                return

        self.place_left = False

        # remove what is already in the buffer
        x, y, idx = x[place_left:], y[place_left:], idx[place_left:]

        indices = torch.FloatTensor(x.size(0)).to(x.device).uniform_(0, self.n_seen_so_far).long()
        valid_indices = (indices < self.bx.size(0))

        idx_new_data = valid_indices.nonzero().squeeze(-1)
        idx_buffer   = indices[idx_new_data]

        self.n_seen_so_far += x.size(0)

        if idx_buffer.numel() == 0 and overwrite:
            return

        # perform overwrite op
        if overwrite:
            self.bx[idx_buffer] = x[idx_new_data]
            self.by[idx_buffer] = y[idx_new_data]
            self.bt[idx_buffer] = t
            self.bidx[idx_buffer] = idx[idx_new_data]
        else:
            del_new_data = (~valid_indices).nonzero().squeeze(-1) + self.bx.size(0)

            """ instead we concatenate, and we will remove later! """
            self.bx = torch.cat((self.bx, x))
            self.by = torch.cat((self.by, y))
            self.bt = torch.cat((self.bt, torch.zeros_like(y).fill_(t)))
            self.bidx = torch.cat((self.bidx, idx))

            assert self.by.size() == self.bt.size(), pdb.set_trace()

            keep = torch.ones_like(self.by)
            keep[idx_buffer] = 0
            keep[del_new_data] = 0
            self.keep = keep.bool()

            extra = keep.sum() - self.args.mem_size
            if extra > 0:
                # print(f'extra : {extra}')
                probs = keep.float() / keep.sum()
                also = torch.multinomial(probs, extra)
                keep[also] = 0

            assert keep.sum() == self.args.mem_size, pdb.set_trace()

            # TODO: remove 122 for this one
            # self.keep = keep.bool()

        if save_logits:
            self.logits[idx_buffer] = logits[idx_new_data]


    def balance_memory(self):
        if self.keep is None:
            return

        self.bx = self.bx[self.keep]
        self.by = self.by[self.keep]
        self.bt = self.bt[self.keep]
        self.bidx = self.bidx[self.keep]

        self.keep = None


    def fetch_pos_neg_samples(self, label, task, idx, data=None, task_free=True):
        # a sample is uniquely identifiable using `task` and `idx`

        if type(task) == int:
            task = torch.LongTensor(label.size(0)).to(label.device).fill_(task)

        same_label = label.view(1, -1) == self.by.view(-1, 1)    # buf_size x label_size
        same_task  = task.view(1, -1)  == self.bt.view(-1, 1)    # buf_size x label_size
        same_idx   = idx.view(1, -1)   == self.bidx.view(-1, 1)  # buf_size x label_size
        same_ex    = same_task & same_idx

        task_labels = label.unique()
        real_same_task = same_task

        # TASK FREE METHOD : instead of using the task ID, we'll use labels in
        # the current batch to mimic task
        if task_free:
            same_task = torch.zeros_like(same_task)

            for label_ in task_labels:
                label_exp = label_.view(1, -1).expand_as(same_task)
                same_task = same_task | (label_exp == self.by.view(-1, 1))

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
        xx = 1

        #import ipdb; ipdb.set_trace()
        return self.bx[pos_idx], \
               self.bx[neg_idx_same_t], \
               self.bx[neg_idx_diff_t], \
               is_invalid


    def sample(self, amt, exclude_task=None, exclude_labels=None, ret_ind=False, aug=False):
        assert not (exclude_task is not None and exclude_labels is not None)

        if exclude_task is not None:
            valid_indices = (self.t != exclude_task)
            valid_indices = valid_indices.nonzero().squeeze()
            bx = self.bx[valid_indices]
            by = self.by[valid_indices]
            bt = self.bt[valid_indices]
            bidx = self.bidx[valid_indices]
        elif exclude_labels is not None:
            # all true tensor
            valid_indices = self.bt > -1
            for label in exclude_labels:
                valid_indices = valid_indices & (self.by != label)

            valid_indices = valid_indices.nonzero().squeeze()
            bx = self.bx[valid_indices]
            by = self.by[valid_indices]
            bt = self.bt[valid_indices]
            bidx = self.bidx[valid_indices]
        else:
            bx = self.bx[:self.current_index]
            by = self.by[:self.current_index]
            bt = self.bt[:self.current_index]
            bidx = self.bidx[:self.current_index]

        if bx.size(0) < amt:
            if ret_ind:
                return bx, by, bt, bidx
            else:
                return bx, by, bt
        else:
            indices = torch.from_numpy(np.random.choice(bx.size(0), amt, replace=False))
            indices = indices.to(self.bx.device)

            if aug:
                # this is sort of wrong cause its already normalized
                # needs to be fixed somehow (unnormalized and renormanlized in th e end?>)

                transform = nn.Sequential(
                        kornia.augmentation.RandomCrop(size=self.input_size[1:],padding=4),
                    kornia.augmentation.RandomHorizontalFlip()
                )
                ret = transform(bx[indices])
            else:
                ret = bx[indices]
            if ret_ind:
                return ret, by[indices], bt[indices], bidx[indices]
            else:
                return ret, by[indices], bt[indices]

    def split(self, amt):
        indices = torch.randperm(self.current_index).to(self.args.device)
        return indices[:amt], indices[amt:]


def get_cifar_buffer(args, hH=8, gen=None):
    args.input_size = (hH, hH)
    args.gen = True

    return Buffer(args, gen=gen)
