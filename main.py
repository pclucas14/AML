import time
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from data   import *
from utils  import get_logger, get_temp_logger, logging_per_task, sho_
from buffer import Buffer
from copy   import deepcopy
from pydoc  import locate
from model  import ResNet18, normalize
from methods import *
# from old_methods import get_method

# Arguments
# -----------------------------------------------------------------------------------------

METHODS = ['icarl', 'er', 'mask', 'er_aml', 'er_aml_triplet', 'mir', 'iid', 'iid++', 'icarl', 'er_multihead', 'der', 'agem']
DATASETS = ['split_cifar10', 'split_cifar100', 'miniimagenet']

parser = argparse.ArgumentParser()

""" optimization (fixed across all settings) """
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--buffer_batch_size', type=int, default=10)

# choose your weapon
parser.add_argument('-m','--method', type=str, default='er', choices=METHODS)

""" data """
parser.add_argument('--download', type=int, default=0)
parser.add_argument('--n_workers', type=int, default=8)
parser.add_argument('--data_root', type=str, default='../cl-pytorch/data')
parser.add_argument('--dataset', type=str, default='split_cifar10', choices=DATASETS)

""" setting """
parser.add_argument('--n_iters', type=int, default=1)
parser.add_argument('--n_tasks', type=int, default=-1)
parser.add_argument('--task_free', type=int, default=0)
parser.add_argument('--use_augmentations', type=int, default=0)
parser.add_argument('--samples_per_task', type=int, default=-1)
parser.add_argument('--mem_size', type=int, default=20, help='controls buffer size')

""" logging """
parser.add_argument('--exp_name', type=str, default='tmp')
parser.add_argument('--wandb_project', type=str, default='er_imbalance')
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

parser.add_argument('--old', action='store_true')

args = parser.parse_args()

if args.method in ['iid', 'iid++']:
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
train_loader, val_loader, test_loader  = get_data(args)

if args.wandb_log != 'off':
    import wandb
    wandb.init(project=args.wandb_project, name=args.exp_name, config=args)
    wandb.config.update(args)
else:
    wandb = None

args.mem_size = args.mem_size * args.n_classes

LOG = get_logger(['cls_loss', 'acc'], n_tasks=args.n_tasks)

# Eval model
# -----------------------------------------------------------------------------------------
@torch.no_grad()
def eval_agent(agent, loader, task, mode='valid'):
    agent.eval()

    for task_t in range(task + 1):
        LOG_temp = get_temp_logger(None, ['cls_loss', 'acc'])

        loader.sampler.set_task(task_t)

        # iterate over samples from task
        for i, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)

            logits = agent.predict(data)
            pred = logits.max(1)[1]

            LOG_temp['acc'] += [pred.eq(target).float().mean().item()]

        logging_per_task(wandb, LOG, mode, 'acc', task, task_t,
            np.round(np.mean(LOG_temp['acc']),2))

    print(LOG[mode]['acc'][:task+1, :task+1])

    if wandb is not None:
        wandb.log({mode + '_anytime_acc_avg': LOG[mode]['acc'][0:task + 1, task].mean()})
        wandb.log({mode + '_anytime_last_acc': LOG[mode]['acc'][task, task]})


# Train the model
# -----------------------------------------------------------------------------------------

# CLASSIFIER
model = ResNet18(
        args.n_classes,
        nf=20,
        input_size=args.input_size,
        dist_linear=args.method in ['er_aml', 'er_aml_triplet', 'mask']
        )

model = model.to(device)
model.train()

if args.old:
    opt = torch.optim.SGD(model.parameters(), lr=args.lr)

buffer = Buffer(capacity=args.mem_size).to(device)

print("number of classifier parameters:",
        sum([np.prod(p.size()) for p in model.parameters()]))
print("buffer parameters: ", np.prod(buffer.bx.size()))

# Build Method wrapper
if args.old:
    agent = get_method(args.method)(model, buffer, args)
else:
    # import methods
    if args.method == 'er':
        agent = ER
    elif args.method == 'mask':
        agent = ER_ACE
    elif args.method == 'er_aml_triplet':
        agent = ER_AML_Triplet
    elif args.method == 'er_aml':
        agent = ER_AML
    elif args.method == 'mir':
        agent = MIR
    elif args.method == 'icarl':
        agent = ICARL
    elif args.method == 'agem':
        agent = AGEM

    agent = agent(model, buffer, args)

#----------
# Task Loop
for task in range(args.n_tasks):

    # set task
    train_loader.sampler.set_task(task)

    n_seen = 0
    agent.train()

    #---------------
    # Minibatch Loop

    for i, (x,y) in enumerate(train_loader):

        if n_seen > args.samples_per_task > 0: break

        x = x.to(device)
        y = y.to(device)

        if i==0:
            print('\nTask #{} --> Train Classifier\n'.format(task))

        #---------------
        # Iteration Loop
        inc_data = {'x': x, 'y': y, 't': task}

        for it in range(args.n_iters):

            if args.old:
                aa, bb = agent.observe(inc_data)

                opt.zero_grad()
                (aa + bb).backward()
                opt.step()
            else:
                agent.observe(inc_data)

        n_seen += x.size(0)

        if args.old:
            buffer.add(inc_data)

    # eval_agent(agent, val_loader, task, mode='valid')
    eval_agent(agent, test_loader, task, mode='test')


# final run results
print('\nFinal Results\n')

for mode in ['valid','test']:
    final_accs = LOG[mode]['acc'][:,task]
    logging_per_task(wandb, LOG, mode, 'final_acc', task,
        value=np.round(np.mean(final_accs),2))
    best_acc = np.max(LOG[mode]['acc'], 1)
    final_forgets = best_acc - LOG[mode]['acc'][:,task]
    logging_per_task(wandb, LOG, mode, 'final_forget', task,
        value=np.round(np.mean(final_forgets[:-1]),2))

    LOG[mode]['last_task_acc'] = np.diag(LOG[mode]['acc']).mean()
    LOG[mode]['allbutfirst_tasks_acc'] = np.mean(LOG[mode]['acc'][:task + 1, task][:-1])

    print('\n{}:'.format(mode))
    print('final accuracy: {}'.format(final_accs))
    print('average: {}'.format(LOG[mode]['final_acc']))
    print('final forgetting: {}'.format(final_forgets))
    print('average: {}\n'.format(LOG[mode]['final_forget']))

    if wandb is not None:
        for task in range(args.n_tasks):
            wandb.log(
		{f'{mode}_anytime_acc_avg_all': LOG[mode]['acc'][:task+1, task].mean(),
            	 f'{mode}_last_acc_avg_all': LOG[mode]['acc'][task,task].mean(),
		 f'{mode}final_acc_avg':final_accs.mean(),
                 f'{mode}final_forget_avg':final_forgets.mean(),
                 # f'{mode}final_last_task_acc':final_last_task_acc,
                 # f'{mode}final_allbutfirst_tasks_acc':final_allbutfirst_tasks_acc
                 })
