import numpy as np
import math
from copy import deepcopy
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import get_grad_vector, get_future_step_parameters


#----------
# Functions
dist_kl = lambda y, t_s : F.kl_div(F.log_softmax(y, dim=-1), F.softmax(t_s, dim=-1), reduction='mean') * y.size(0)
# this returns -entropy
entropy_fn = lambda x : torch.sum(F.softmax(x, dim=-1) * F.log_softmax(x, dim=-1), dim=-1)

cross_entropy = lambda y, t_s : -torch.sum(F.log_softmax(y, dim=-1)*F.softmax(t_s, dim=-1),dim=-1).mean()
mse = torch.nn.MSELoss()

def retrieve_replay_update(args, model, opt, input_x, input_y, buffer, task, loader = None, rehearse=True):
    """ finds buffer samples with maxium interference """

    '''
    ER - MIR and regular ER
    '''


    updated_inds = None

    hid = model.return_hidden(input_x)

    logits = model.linear(hid)
    if args.multiple_heads:
        logits = logits.masked_fill(loader.dataset.mask == 0, -1e9)
    loss_a = F.cross_entropy(logits, input_y, reduction='none')
    loss = (loss_a).sum() / loss_a.size(0)

    opt.zero_grad()
    loss.backward()

    if not rehearse:
        opt.step()
        return model

    if args.method == 'mir_replay':

        exclude_task = None if args.task_free else task

        bx, by, bt, subsample = buffer.sample(args.subsample, exclude_task=exclude_task, ret_ind=True)
        grad_dims = []
        for param in model.parameters():
            grad_dims.append(param.data.numel())
        grad_vector = get_grad_vector(args, model.parameters, grad_dims)
        model_temp = get_future_step_parameters(model, grad_vector,grad_dims, lr=args.lr)

        with torch.no_grad():
            logits_track_pre = model(bx)
            buffer_hid = model_temp.return_hidden(bx)
            logits_track_post = model_temp.linear(buffer_hid)

            if args.multiple_heads:
                mask = torch.zeros_like(logits_track_post)
                mask.scatter_(1, loader.dataset.task_ids[bt], 1)
                assert mask.nelement() // mask.sum() == args.n_tasks
                logits_track_post = logits_track_post.masked_fill(mask == 0, -1e9)
                logits_track_pre = logits_track_pre.masked_fill(mask == 0, -1e9)

            pre_loss = F.cross_entropy(logits_track_pre, by , reduction="none")
            post_loss = F.cross_entropy(logits_track_post, by , reduction="none")
            scores = post_loss - pre_loss

            EN_logits = entropy_fn(logits_track_pre)
            if args.compare_to_old_logits:
                old_loss = F.cross_entropy(buffer.logits[subsample], by,reduction="none")

                updated_mask = pre_loss < old_loss
                updated_inds = updated_mask.data.nonzero().squeeze(1)
                scores = post_loss - torch.min(pre_loss, old_loss)

            all_logits = scores
            big_ind = all_logits.sort(descending=True)[1][:args.buffer_batch_size]

            idx = subsample[big_ind]

        mem_x, mem_y, logits_y, b_task_ids = bx[big_ind], by[big_ind], buffer.logits[idx], bt[big_ind]

    else:
        mem_x, mem_y, bt = buffer.sample(args.buffer_batch_size, exclude_task=None if args.task_free else task)

    logits_buffer = model(mem_x)
    if args.multiple_heads:
        mask = torch.zeros_like(logits_buffer)
        mask.scatter_(1, loader.dataset.task_ids[b_task_ids], 1)
        assert mask.nelement() // mask.sum() == args.n_tasks
        logits_buffer = logits_buffer.masked_fill(mask == 0, -1e9)
    F.cross_entropy(logits_buffer, mem_y).backward()

    if updated_inds is not None:
        buffer.logits[subsample[updated_inds]] = deepcopy(logits_track_pre[updated_inds])
    opt.step()
    return model

