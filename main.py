import time
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict as OD

from logger    import Logger
from copy      import deepcopy
from data.base import *
from utils     import get_logger, get_temp_logger, logging_per_task, sho_
from copy      import deepcopy
from pydoc     import locate
from model     import ResNet18, normalize
from methods   import *

# Arguments
# -----------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()

""" optimization (fixed across all settings) """
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--buffer_batch_size', type=int, default=10)

# choose your weapon
parser.add_argument('-m','--method', type=str, default='er', choices=METHODS.keys())

""" data """
parser.add_argument('--H', type=int, default=None)
parser.add_argument('--download', type=int, default=0)
parser.add_argument('--n_workers', type=int, default=8)
parser.add_argument('--data_root', type=str, default='../cl-pytorch/data')
parser.add_argument('--dataset', type=str, default='split_cifar10', choices=DATASETS)

""" setting """
parser.add_argument('--n_iters', type=int, default=1)
parser.add_argument('--n_tasks', type=int, default=-1)
parser.add_argument('--task_free', type=int, default=0)
parser.add_argument('--use_augs', type=int, default=0)
parser.add_argument('--samples_per_task', type=int, default=-1)
parser.add_argument('--mem_size', type=int, default=20, help='controls buffer size')
parser.add_argument('--validation', type=int, default=1)
parser.add_argument('--eval_every', type=int, default=-1)

""" logging """
parser.add_argument('--exp_name', type=str, default='tmp')
parser.add_argument('--wandb_project', type=str, default='online_cl')
parser.add_argument('--wandb_log', type=str, default='off', choices=['off', 'online'])

""" HParams """
parser.add_argument('--lr', type=float, default=0.1)

# ER-AML hparams
parser.add_argument('--margin', type=float, default=0.2)
parser.add_argument('--buffer_neg', type=float, default=0)
parser.add_argument('--incoming_neg', type=float, default=2.0)
parser.add_argument('--supcon_temperature', type=float, default=0.2)

# ICARL hparams
parser.add_argument('--distill_coef', type=float, default=1.)

# DER params
parser.add_argument('--alpha', type=float, default=.1)
parser.add_argument('--beta', type=float, default=.5)

# MIR params
parser.add_argument('--subsample', type=int, default=50)
parser.add_argument('--mir_head_only', type=int, default=0)

# CoPE params
parser.add_argument('--momentum', type=float, default=0.99)
parser.add_argument('--cope_temperature', type=float, default=0.1)

args = parser.parse_args()

if args.method in ['iid', 'iid++']:
    raise ValueError

    print('overwriting args for iid setup')
    args.n_tasks = 1
    args.mem_size = 1


# Obligatory overhead
# -----------------------------------------------------------------------------------------

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

# make dataloaders
train_tf, train_loader, val_loader, test_loader  = get_data_and_tfs(args)
train_tf.to(device)

logger = Logger(args)
args.mem_size = args.mem_size * args.n_classes


# Eval model
# -----------------------------------------------------------------------------------------

@torch.no_grad()
def eval_agent(agent, loader, task, mode='valid'):
    global logger

    agent.eval()

    accs = np.zeros(shape=(loader.sampler.n_tasks,))

    for task_t in range(task + 1):

        n_ok, n_total = 0, 0
        loader.sampler.set_task(task_t)

        # iterate over samples from task
        for i, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)

            logits = agent.predict(data)
            pred   = logits.max(1)[1]

            n_ok    += pred.eq(target).sum().item()
            n_total += data.size(0)

        accs[task_t] = n_ok / n_total * 100


    avg_acc = np.mean(accs[:task + 1])
    print('\n', '\t'.join([str(int(x)) for x in accs]), f'\tAvg Acc: {avg_acc:.2f}')

    logger.log_scalars({
        f'{mode}/anytime_last_acc': accs[task],
        f'{mode}/anytime_acc_avg_seen': avg_acc,
        f'{mode}/anytime_acc_avg_all': np.mean(accs),
    })

    return accs


# Train the model
# -----------------------------------------------------------------------------------------

# CLASSIFIER
model = ResNet18(
        args.n_classes,
        nf=20,
        input_size=args.input_size,
        dist_linear='ace' in args.method or 'aml' in args.method
        )

model = model.to(device)
model.train()

agent = METHODS[args.method](model, logger, train_tf, args)
n_params = sum(np.prod(p.size()) for p in model.parameters())

print("number of classifier parameters:", n_params)

eval_accs = []
if args.validation:
    mode = 'valid'
    eval_loader = val_loader
else:
    mode = 'test'
    eval_laoder = test_loader

#----------
# Task Loop
for task in range(args.n_tasks):

    # set task
    train_loader.sampler.set_task(task)

    n_seen = 0
    agent.train()

    #---------------
    # Minibatch Loop

    print('\nTask #{} --> Train Classifier\n'.format(task))
    for i, (x,y) in enumerate(train_loader):
        print(f'{i} / {len(train_loader)}', end='\r')

        if n_seen > args.samples_per_task > 0: break

        x = x.to(device)
        y = y.to(device)

        inc_data = {'x': x, 'y': y, 't': task}

        agent.observe(inc_data)

        n_seen += x.size(0)

    # always eval at end of tasks
    eval_accs  += [eval_agent(agent, eval_loader, task, mode=mode)]


# ----- Final Results ----- #

accs    = np.stack(eval_accs).T
avg_acc = accs[:, -1].mean()
avg_fgt = (accs.max(1) - accs[:, -1])[:-1].mean()

print('\nFinal Results\n')
logger.log_matrix(f'{mode}_acc', accs)
logger.log_scalars({
    f'{mode}/avg_acc': avg_acc,
    f'{mode}/avg_fgt': avg_fgt,
    'train/n_samples': n_seen,
    'metrics/model_n_bits': n_params * 32,
    'metrics/cost': agent.cost,
    'metrics/one_sample_flop': agent.one_sample_flop,
    'metrics/buffer_n_bits': agent.buffer.n_bits()
})

logger.close()
