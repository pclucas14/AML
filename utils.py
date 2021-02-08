import torch
import torch.nn.functional as F
import numpy as np
import copy
import pdb
from collections import OrderedDict as OD
from collections import defaultdict as DD

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

''' LOG '''
def logging_per_task(wandb, log, mode, metric, task=0, task_t=0, value=0):
    if 'final' in metric:
        log[mode][metric] = value
    else:
        log[mode][metric][task_t, task] = value

    if wandb is not None:
        if 'final' in metric:
            wandb.log({mode+metric:value}) #, step=run)

def print_(log, mode, task):
    to_print = mode + ' ' + str(task) + ' '
    for name, value in log.items():
        # only print acc for now
        if len(value) > 0:
            name_ = name + ' ' * (12 - len(name))
            value = sum(value) / len(value)

            if 'acc' in name or 'gen' in name:
                to_print += '{}\t {:.4f}\t'.format(name_, value)

    print(to_print)

def get_logger(names, n_tasks=None):
    log = OD()
    log.print_ = lambda a, b: print_(log, a, b)
    log = {}
    for mode in ['train','valid','test']:
        log[mode] = {}
        for name in names:
            log[mode][name] = np.zeros([n_tasks,n_tasks])

        log[mode]['final_acc'] = 0.
        log[mode]['final_forget'] = 0.

    return log

def get_temp_logger(exp, names):
    log = OD()
    log.print_ = lambda a, b: print_(log, a, b)
    for name in names: log[name] = []
    return log


import collections

import numpy as np
import torch

def sho_(x, nrow=8):
    x = x * .5 + .5
    from torchvision.utils import save_image
    from PIL import Image
    if x.ndim == 5:
        nrow=x.size(1)
        x = x.reshape(-1, *x.shape[2:])

    save_image(x, 'tmp.png', nrow=nrow)
    Image.open('tmp.png').show()
