import torch
import torch.nn.functional as F
import numpy as np
import copy
import pdb
from collections import OrderedDict as OD
from collections import defaultdict as DD

torch.random.manual_seed(0)



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target, mask=None):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        if mask is not None:
            import ipdb; ipdb.set_trace()
        else:
            loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

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

def get_grad_vector(args, pp, grad_dims):
    """
     gather the gradients in one vector
    """
    grads = torch.Tensor(sum(grad_dims))
    if args.cuda: grads = grads.cuda()

    grads.fill_(0.0)
    cnt = 0
    for param in pp():
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

''' Others '''
def onehot(t, num_classes, device='cpu'):
    """
    convert index tensor into onehot tensor
    :param t: index tensor
    :param num_classes: number of classes
    """
    return torch.zeros(t.size()[0], num_classes).to(device).scatter_(1, t.view(-1, 1), 1)

def distillation_KL_loss(y, teacher_scores, T, scale=1, reduction='batchmean'):
    """Computes the distillation loss (cross-entropy).
       xentropy(y, t) = kl_div(y, t) + entropy(t)
       entropy(t) does not contribute to gradient wrt y, so we skip that.
       Thus, loss value is slightly different, but gradients are correct.
       \delta_y{xentropy(y, t)} = \delta_y{kl_div(y, t)}.
       scale is required as kl_div normalizes by nelements and not batch size.
    """
    return F.kl_div(F.log_softmax(y / T, dim=1), F.softmax(teacher_scores / T, dim=1),
            reduction=reduction) * scale

def naive_cross_entropy_loss(input, target, size_average=True):
    """
    in PyTorch's cross entropy, targets are expected to be labels
    so to predict probabilities this loss is needed
    suppose q is the target and p is the input
    loss(p, q) = -\sum_i q_i \log p_i
    """
    assert input.size() == target.size()
    input = torch.log(F.softmax(input, dim=1).clamp(1e-5, 1))
    # input = input - torch.log(torch.sum(torch.exp(input), dim=1)).view(-1, 1)
    loss = - torch.sum(input * target)
    return loss / input.size()[0] if size_average else loss

def compute_offsets(task, nc_per_task, is_cifar):
    """
        Compute offsets for cifar to determine which
        outputs to select for a given task.
    """
    if is_cifar:
        offset1 = task * nc_per_task
        offset2 = (task + 1) * nc_per_task
    else:
        offset1 = 0
        offset2 = nc_per_task
    return offset1, offset2

def out_mask(t, nc_per_task, n_outputs):
    # make sure we predict classes within the current task
    offset1 = int(t * nc_per_task)
    offset2 = int((t + 1) * nc_per_task)
    if offset1 > 0:
        output[:, :offset1].data.fill_(-10e10)
    if offset2 < self.n_outputs:
        output[:, offset2:n_outputs].data.fill_(-10e10)

class Reshape(torch.nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape
    def forward(self, input):
        return input.view(input.size(0), *self.shape)

''' LOG '''
def logging_per_task(wandb, log, run, mode, metric, task=0, task_t=0, value=0):
    if 'final' in metric:
        log[run][mode][metric] = value
    else:
        log[run][mode][metric][task_t, task] = value

    if wandb is not None:
        if 'final' in metric:
            wandb.log({mode+metric:value}, step=run)

def print_(log, mode, task):
    to_print = mode + ' ' + str(task) + ' '
    for name, value in log.items():
        # only print acc for now
        if len(value) > 0:
            name_ = name + ' ' * (12 - len(name))
            value = sum(value) / len(value)

            if 'acc' in name or 'gen' in name:
                to_print += '{}\t {:.4f}\t'.format(name_, value)
                # print('{}\t {}\t task {}\t {:.4f}'.format(mode, name_, task, value))

    print(to_print)

def get_logger(names, n_runs=1, n_tasks=None):
    log = OD()
    #log = DD()
    log.print_ = lambda a, b: print_(log, a, b)
    for i in range(n_runs):
        log[i] = {}
        for mode in ['train','valid','test']:
            log[i][mode] = {}
            for name in names:
                log[i][mode][name] = np.zeros([n_tasks,n_tasks])

            log[i][mode]['final_acc'] = 0.
            log[i][mode]['final_forget'] = 0.

    return log

def get_temp_logger(exp, names):
    log = OD()
    log.print_ = lambda a, b: print_(log, a, b)
    for name in names: log[name] = []
    return log


import collections

import numpy as np
import torch
# from loguru import logger


class SelectiveBackPropagation:
    """
    Selective_Backpropagation from paper Accelerating Deep Learning by Focusing on the Biggest Losers
    https://arxiv.org/abs/1910.00762v1
    Without:
            ...
            criterion = nn.CrossEntropyLoss(reduction='none')
            ...
            for x, y in data_loader:
                ...
                y_pred = model(x)
                loss = criterion(y_pred, y).mean()
                loss.backward()
                ...
    With:
            ...
            criterion = nn.CrossEntropyLoss(reduction='none')
            selective_backprop = SelectiveBackPropagation(
                                    criterion,
                                    lambda loss : loss.mean().backward(),
                                    model,
                                    batch_size,
                                    epoch_length=len(data_loader),
                                    loss_selection_threshold=False)
            ...
            for x, y in data_loader:
                ...
                with torch.no_grad():
                    y_pred = model(x)
                not_reduced_loss = criterion(y_pred, y)
                selective_backprop.selective_back_propagation(not_reduced_loss, x, y)
                ...
    """
    def __init__(self, batch_size, epoch_length, loss_selection_threshold=False):
        """
        Usage:
        ```
        criterion = nn.CrossEntropyLoss(reduction='none')
        selective_backprop = SelectiveBackPropagation(
                                    lambda loss : loss.mean().backward(),
                                    model,
                                    batch_size,
                                    epoch_length=len(data_loader),
                                    loss_selection_threshold=False)
        ```
        :param compute_losses_func: the loss function which output a tensor of dim [batch_size] (no reduction to apply).
        Example: `compute_losses_func = nn.CrossEntropyLoss(reduction='none')`
        :param update_weights_func: the reduction of the loss and backpropagation. Example: `update_weights_func =
        lambda loss : loss.mean().backward()`
        :param optimizer: your optimizer object
        :param model: your model object
        :param batch_size: number of images per batch
        :param epoch_length: the number of batch per epoch
        :param loss_selection_threshold: default to False. Set to a float value to select all images with with loss
        higher than loss_selection_threshold. Do not change behavior for loss below loss_selection_threshold.
        """

        self.loss_selection_threshold = loss_selection_threshold
        self.batch_size = batch_size

        self.loss_hist = collections.deque([], maxlen=batch_size*epoch_length)
        self.selected_inputs, self.selected_targets = [], []

    def selective_back_propagation_idx(self, loss_per_img,stats_only=False):
        effective_batch_loss = None

        cpu_losses = loss_per_img.detach().clone().cpu()
        self.loss_hist.extend(cpu_losses.tolist())
        if stats_only:
            return
        np_cpu_losses = cpu_losses.numpy()
        selection_probabilities = self._get_selection_probabilities(np_cpu_losses)

        selection = selection_probabilities > np.random.random(*selection_probabilities.shape)

        if self.loss_selection_threshold:
            higher_thres = np_cpu_losses > self.loss_selection_threshold
            selection = np.logical_or(higher_thres, selection)

        return selection

    def _get_selection_probabilities(self, loss):
        percentiles = self._percentiles(self.loss_hist, loss)
        return percentiles ** 2

    def _percentiles(self, hist_values, values_to_search):
        # TODO Speed up this again. There is still a visible overhead in training.
        hist_values, values_to_search = np.asarray(hist_values), np.asarray(values_to_search)

        percentiles_values = np.percentile(hist_values, range(100))
        sorted_loss_idx = sorted(range(len(values_to_search)), key=lambda k: values_to_search[k])
        counter = 0
        percentiles_by_loss = [0] * len(values_to_search)
        for idx, percentiles_value in enumerate(percentiles_values):
            while values_to_search[sorted_loss_idx[counter]] < percentiles_value:
                percentiles_by_loss[sorted_loss_idx[counter]] = idx
                counter += 1
                if counter == len(values_to_search) : break
            if counter == len(values_to_search) : break
        return np.array(percentiles_by_loss)/100