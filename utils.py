import os
import copy
import numpy as np
from collections import OrderedDict as OD
from collections import defaultdict as DD

import torch
import torch.nn as nn
import torch.nn.functional as F

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


# --- MIR utils
''' For MIR '''
def overwrite_grad(pp, new_grad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    cnt = 0
    for param in pp():
        param.grad=torch.zeros_like(param.data)
        beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
        en = sum(grad_dims[:cnt + 1])
        this_grad = new_grad[beg: en].contiguous().view(
            param.data.size())
        param.grad.data.copy_(this_grad)
        cnt += 1

def get_grad_vector(pp, grad_dims):
    """
     gather the gradients in one vector
    """
    grads = torch.zeros(size=(sum(grad_dims),), device=pp[0].device)

    cnt = 0
    for param in pp:
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en].copy_(param.grad.data.view(-1))
        cnt += 1
    return grads

def get_future_step_parameters(this_net, grad_vector, grad_dims, lr=1):
    """
    computes \theta-\delta\theta
    :param this_net:
    :param grad_vector:
    :return:
    """
    new_net=copy.deepcopy(this_net)
    overwrite_grad(new_net.parameters,grad_vector,grad_dims)
    with torch.no_grad():
        for param in new_net.parameters():
            if param.grad is not None:
                param.data=param.data - lr*param.grad.data
    return new_net

def get_grad_dims(self):
    self.grad_dims = []
    for param in self.net.parameters():
        self.grad_dims.append(param.data.numel())

# Taken from
# https://github.com/aimagelab/mammoth/blob/cb9a36d788d6ad051c9eee0da358b25421d909f5/models/gem.py#L34
def store_grad(params, grads, grad_dims):
    """
        This stores parameter gradients of past tasks.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers
    """
    # store the gradients
    grads.fill_(0.0)
    count = 0
    for param in params():
        if param.grad is not None:
            begin = 0 if count == 0 else sum(grad_dims[:count])
            end = np.sum(grad_dims[:count + 1])
            grads[begin: end].copy_(param.grad.data.view(-1))
        count += 1

# Taken from
# https://github.com/aimagelab/mammoth/blob/cb9a36d788d6ad051c9eee0da358b25421d909f5/models/agem.py#L21
def project(gxy: torch.Tensor, ger: torch.Tensor) -> torch.Tensor:
    corr = torch.dot(gxy, ger) / torch.dot(ger, ger)
    return gxy - corr * ger
