import os
import copy
import numpy as np
from collections import OrderedDict as OD
from collections import defaultdict as DD

import torch
import torch.nn as nn
import torch.nn.functional as F

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

# https://github.com/tristandeleu/pytorch-meta/
from torchvision.datasets.utils import _get_confirm_token, _save_response_content

def _quota_exceeded(response: "requests.models.Response"):
    return False
    # See https://github.com/pytorch/vision/issues/2992 for details
    # return "Google Drive - Quota exceeded" in response.text


def download_file_from_google_drive(file_id, root, filename=None, md5=None):
    """Download a Google Drive file from  and place it in root.
    Args:
        file_id (str): id of file to be downloaded
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the id of the file.
        md5 (str, optional): MD5 checksum of the download. If None, do not check
    """
    # Based on https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
    import requests
    url = "https://docs.google.com/uc?export=download"

    root = os.path.expanduser(root)
    if not filename:
        filename = file_id
    fpath = os.path.join(root, filename)

    os.makedirs(root, exist_ok=True)

    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        session = requests.Session()

        response = session.get(url, params={'id': file_id}, stream=True)
        token = _get_confirm_token(response)

        if token:
            params = {'id': file_id, 'confirm': token}
            response = session.get(url, params=params, stream=True)

        if _quota_exceeded(response):
            msg = (
                f"The daily quota of the file {filename} is exceeded and it "
                f"can't be downloaded. This is a limitation of Google Drive "
                f"and can only be overcome by trying again later."
            )
            raise RuntimeError(msg)

        _save_response_content(response, fpath)


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
