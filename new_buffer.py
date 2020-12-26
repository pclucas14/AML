import pdb
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class Buffer(nn.Module):
    def __init__(self, input_size, n_classes, cap, amt=0):
        super().__init__()

        self.input_size = input_size
        self.n_classes  = n_classes

        bx    = torch.FloatTensor(amt, *input_size).fill_(0)
        by    = torch.LongTensor(amt).fill_(0)
        bt    = torch.LongTensor(amt).fill_(0)
        bidx  = torch.LongTensor(amt).fill_(0)

        self.cap = cap
        self.n_samples = amt

        self.register_buffer('bx', bx)
        self.register_buffer('by', by)
        self.register_buffer('bt', bt)
        self.register_buffer('bidx', bidx)

        self.to_one_hot  = lambda x : x.new(x.size(0), n_classes).fill_(0).scatter_(1, x.unsqueeze(1), 1)
        self.arange_like = lambda x : torch.arange(x.size(0)).to(x.device)
        self.shuffle     = lambda x : x[torch.randperm(x.size(0))]

    def expand(self, amt):
        """ used when loading a model from `pth` file and the amt of samples in the buffer don't align """
        self.__init__(self.input_size, self.n_classes, dtype=self.dtype, amt=amt)

    @property
    def x(self):
        return self.bx[:self.n_samples]

    @property
    def y(self):
        return self.to_one_hot(self.by[:self.n_samples])

    @property
    def t(self):
        return self.bt[:self.n_samples]

    @torch.no_grad()
    def add(self, in_x, in_y, in_t, in_idx):

        """ concatenate a sample at the end of the buffer """

        # convert int `in_t` to long tensor
        if type(in_t) == int:
            in_t = torch.LongTensor(in_x.size(0)).to(in_x.device).fill_(in_t)

        if self.bx.size(0) > in_x.size(0):
            swap_idx = torch.randperm(self.bx.size(0))[:in_x.size(0)]

            tmp_x    = self.bx[swap_idx]
            tmp_y    = self.by[swap_idx]
            tmp_t    = self.bt[swap_idx]
            tmp_idx  = self.bidx[swap_idx]

            # overwrite
            self.bx[swap_idx]    = in_x
            self.by[swap_idx]    = in_y
            self.bt[swap_idx]    = in_t
            self.bidx[swap_idx]  = in_idx

            in_x, in_y, in_t, in_idx = tmp_x, tmp_y, tmp_t, tmp_idx

        self.bx     = torch.cat((self.bx, in_x))
        self.by     = torch.cat((self.by, in_y))
        self.bt     = torch.cat((self.bt, in_t))
        self.bidx   = torch.cat((self.bidx, in_idx))

        self.n_samples += in_x.size(0)


    @torch.no_grad()
    def free(self, n_samples=None, idx=None):
        """ free buffer space. Assumes data is shuffled when added"""

        assert n_samples is not None or idx is not None, \
                'must specify amt of points to remove, or specific idx'

        if n_samples is None:
            n_samples = idx.size(0) if idx.ndim > 0 else 0

        if n_samples == 0:
            return 0, 0

        n_samples = int(n_samples)
        assert n_samples <= self.n_samples, pdb.set_trace()

        if idx is not None:
            class_removed = self.y[idx].sum(0)

            idx_to_keep = torch.ones_like(self.by)
            idx_to_keep[idx] = 0
            idx_to_keep = idx_to_keep.nonzero().squeeze(1)

            self.bx = self.bx[idx_to_keep]
            self.by = self.by[idx_to_keep]
            self.bt = self.bt[idx_to_keep]
            self.bidx = self.bidx[idx_to_keep]
        else:
            class_removed = self.y[-n_samples:].sum(0)

            self.bx = self.bx[:-n_samples]
            self.by = self.by[:-n_samples]
            self.bt = self.bt[:-n_samples]
            self.bidx = self.bidx[:-n_samples]

        self.n_samples -= n_samples

        return class_removed, n_samples


    @torch.no_grad()
    def balance_memory(self):

        n_samples = max(0, self.n_samples - self.cap)

        if n_samples == 0:
            return 0, 0

        class_counts = self.y.sum(0)
        """ figure out how many samples per class should be removed """

        # sort classes w.r.t count
        class_count, class_id = torch.sort(class_counts, descending=True)

        gain = torch.zeros_like(class_count)
        gain[1:] = class_count[:-1] - class_count[1:]
        cum_gain = gain.cumsum(0)

        # get class counts for removal
        counts = torch.zeros(n_samples, self.n_classes).to(class_counts.device)

        # don't bother with classes having too few elems to reach n_samples
        valid_idx = cum_gain < n_samples
        counts[cum_gain[valid_idx], torch.arange(self.n_classes)[valid_idx]] = 1

        counts = counts.cumsum(0)
        cum_counts = counts.cumsum(0)
        total_cum_counts = cum_counts.sum(1)

        idx = (total_cum_counts < n_samples).sum()

        to_be_removed_counts = tbr_counts = cum_counts[(idx - 1).clamp_(min=0)]
        missing = int(n_samples - tbr_counts.sum())

        tbr_old = tbr_counts.clone()

        if missing != 0:
            # randomly assign the missing samples to available classes
            n_avail_classes = tbr_counts.nonzero().size(0)
            sample = torch.LongTensor(abs(missing)).random_(0, self.n_classes).to(counts.device)
            sample = sample % n_avail_classes
            tbr_counts[:n_avail_classes] += np.sign(missing) * sample.bincount(minlength=n_avail_classes)

        assert tbr_counts.sum() == n_samples, pdb.set_trace()

        """ remove class specific samples """

        # restore valid order
        tbr_counts = tbr_counts[class_id.sort()[1]]

        # buffer is already in random order, so just remove from the top
        class_total = self.y.cumsum(0)

        #       did we reach cap already?    get actual label
        tbr = ((class_total <= tbr_counts) & self.y.bool()).int() #.sum(0)

        tbr_idx = tbr.sum(1).nonzero().squeeze(-1)

        return self.free(idx=tbr_idx)


    @torch.no_grad()
    def sample(self, amt, exclude_task=None, aug=False):

        if aug:
            raise NotImplementedError

        if amt == 0:
            return self.bx[:0], self.by[:0], self.bt[:0], self.bidx[:0]

        if exclude_task is not None:
            valid_indices = (self.t != exclude_task)
            valid_indices = valid_indices.nonzero().squeeze()
            bx = self.bx[valid_indices]
            by = self.by[valid_indices]
            bt = self.bt[valid_indices]
            bidx = self.bidx[valid_indices]
        else:
            bx, by, bt, bidx = self.bx, self.by, self.bt, self.bidx

        indices = torch.from_numpy(np.random.choice(bx.size(0), amt, replace=False)).to(self.bx.device)

        return bx[indices], by[indices], bt[indices], bidx[indices]


    def fetch_pos_neg_samples(self, label, task, idx):
        # a sample is uniquely identifiable using `task` and `idx`

        if type(task) == int:
            task = torch.LongTensor(label.size(0)).to(label.device).fill_(task)

        same_label = label.view(1, -1) == self.by.view(-1, 1)   # buf_size x label_size
        same_task  = task.view(1, -1)  == self.t.view(-1, 1)    # buf_size x label_size
        same_idx   = idx.view(1, -1)   == self.bidx.view(-1, 1) # buf_size x label_size
        same_ex    = same_task & same_idx

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
            valid_nef_diff_t[:, invalid_idx] = 1

        # easier if invalid_idx is a binary tensor
        is_invalid = torch.zeros_like(label).bool()
        is_invalid[invalid_idx] = 1

        # fetch positive samples
        pos_idx = torch.multinomial(valid_pos.float().T, 1).squeeze(1)
        neg_idx_same_t = torch.multinomial(valid_neg_same_t.float().T, 1).squeeze(1)
        neg_idx_diff_t = torch.multinomial(valid_neg_diff_t.float().T, 1).squeeze(1)

        return self.bx[pos_idx], \
               self.bx[neg_idx_same_t], \
               self.bx[neg_idx_diff_t], \
               is_invalid


if __name__ == '__main__':
    import numpy as np

    INPUT_SIZE = (3, 32, 32)
    N_CLASSES  = 10
    buf = Buffer(INPUT_SIZE, N_CLASSES, cap=1000)

    for task in range(5):
        total = 0
        for i in range(100):
            B = np.random.randint(100)

            in_x = torch.FloatTensor(size=(B, ) + INPUT_SIZE).normal_()
            in_y = torch.FloatTensor(B).uniform_(0, N_CLASSES).long()
            in_t = torch.zeros_like(in_y) + task
            in_idx = torch.arange(total, total + B).to(in_y)

            total += B
            data = {'x': in_x, 'y': in_y, 't': in_t, 'bidx': in_idx}

            buf.add(data)
            print(f'{buf.n_samples}\t{buf.y.sum(0)}')

            if task > 0:
                out = buf.fetch_pos_neg_samples(in_y, in_t, in_idx)

            buf.balance_memory(np.random.randint(100))
            print(f'{buf.n_samples}\t{buf.y.sum(0)}\n\n')

