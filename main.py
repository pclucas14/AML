import time
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict as OD
import wandb

from utils     import sho_, load_best_args
from logger    import Logger
from copy      import deepcopy
from data.base import *
from copy      import deepcopy
from pydoc     import locate
from model     import ResNet18, normalize
from methods   import *

torch.set_num_threads(4)

# Arguments
# -----------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()

""" optimization (fixed across all settings) """
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--buffer_batch_size', type=int, default=10)

# choose your weapon
parser.add_argument('-m','--method', type=str, default='er', choices=METHODS.keys())

""" data """
parser.add_argument('--download', type=int, default=0)
parser.add_argument('--data_root', type=str, default='../cl-pytorch/data')
parser.add_argument('--dataset', type=str, default='cifar10', choices=DATASETS)
parser.add_argument('--smooth', type=int, default=0)

parser.add_argument('--nf', type=int, default=20)

""" setting """
parser.add_argument('--n_iters', type=int, default=1)
parser.add_argument('--n_tasks', type=int, default=-1)
parser.add_argument('--task_free', type=int, default=0)
parser.add_argument('--use_augs', type=int, default=0)
parser.add_argument('--samples_per_task', type=int, default=-1)
parser.add_argument('--mem_size', type=int, default=20, help='controls buffer size')
parser.add_argument('--eval_every', type=int, default=-1)
parser.add_argument('--run', type=int, default=0)
parser.add_argument('--validation', type=int, default=1)
parser.add_argument('--load_best_args', type=int, default=0)

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

# ICARL hparams / SS-IL
parser.add_argument('--distill_coef', type=float, default=0.)

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

if args.load_best_args:
    load_best_args(args)

if args.method in ['iid', 'iid++']:
    print('overwriting args for iid setup')
    args.n_tasks = 1
    args.mem_size = 0


# Obligatory overhead
# -----------------------------------------------------------------------------------------

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# make dataloaders
train_tf, train_loader, val_loader, test_loader  = get_data_and_tfs(args)
train_tf.to(device)

logger = Logger(args)
args.mem_size = args.mem_size * args.n_classes

# for iid methods
args.train_loader = train_loader

# CLASSIFIER
model = ResNet18(
        args.n_classes,
        nf=args.nf,
        input_size=args.input_size,
        dist_linear=True #'ace' in args.method or 'aml' in args.method
        )

model = model.to(device)
model.train()

wandb.init(project='ace_drift', name=args.exp_name, config=args)

agent = METHODS[args.method](model, logger, train_tf, args)
n_params = sum(np.prod(p.size()) for p in model.parameters())
buffer = agent.buffer
not_added = buffer.old_buffer

print(model)
print("number of classifier parameters:", n_params)

eval_accs = []
if args.validation:
    mode = 'valid'
    eval_loader = val_loader
else:
    mode = 'test'
    eval_loader = test_loader

test_xs, test_ys = [], []
for task in range(args.n_tasks):
    val_loader.sampler.set_task(task)
    aa, bb = next(iter(val_loader))
    test_xs += [aa.to(device)]
    test_ys += [bb.to(device)]


def measure_drift(model, x, y, proto=None):
    model.eval()
    with torch.no_grad():
        hid = model.return_hidden(x)
        hid = F.normalize(hid, dim=-1, p=2)

        if proto is None:
            proto=model.linear.L.weight

        try:
            proto = F.normalize(proto, dim=-1, p=2)
        except:
            proto = F.normalize(proto, dim=-1, p=2)

        cosine = hid.matmul(proto.T)
        cosine = cosine[torch.arange(x.size(0)), y]

    model.train()
    return cosine.mean()


@torch.no_grad()
def knn_classifier(train_features, train_labels, test_features, test_labels, k=20, T=0.1, num_classes=10):

    # normalize
    train_features = nn.functional.normalize(train_features, dim=1, p=2)
    test_features = nn.functional.normalize(test_features, dim=1, p=2)

    top1, top5, total = 0.0, 0.0, 0
    train_features = train_features.t()
    num_test_images, num_chunks = test_labels.shape[0], 100
    imgs_per_chunk = num_test_images // num_chunks
    retrieval_one_hot = torch.zeros(k, num_classes).cuda()
    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = test_features[
            idx : min((idx + imgs_per_chunk), num_test_images), :
        ]
        targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(k, largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(T).exp_()
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        top5 = top5 + correct.narrow(1, 0, 5).sum().item()
        total += targets.size(0)
    top1 = top1 * 100.0 / total
    top5 = top5 * 100.0 / total
    return top1, top5


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

            if device == 'cuda':
                data   = data.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

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

#----------
# Task Loop
for task in range(args.n_tasks):

    # set task
    train_loader.sampler.set_task(task)

    n_seen = 0
    unique = 0
    agent.train()
    start = time.time()

    #---------------
    # Minibatch Loop

    print('\nTask #{} --> Train Classifier\n'.format(task))
    for i, (x,y) in enumerate(train_loader):

        to_log = {}
        if (task + i) > 100:
            with torch.no_grad():
                model.eval()
                knns_tr, knns_buf = [], []

                # make sure not added buffer has the same sample dist
                counts = buffer.by.bincount()
                idxs = []
                for it, count in enumerate(counts):
                    idx = torch.where(not_added.by == it)[0][:count]
                    idxs += [idx]

                idxs = torch.cat(idxs)
                tr_x, tr_y = not_added.bx[idxs], not_added.by[idxs]
                bf_x, bf_y = buffer.bx, buffer.by

                assert (tr_y.bincount() == bf_y.bincount()).all(), pdb.set_trace()

                tr_hid = F.normalize(model.return_hidden(tr_x), p=2, dim=-1)
                bf_hid = F.normalize(model.return_hidden(bf_x), p=2, dim=-1)

                for test_task in range(task + 1):
                    feat = model.return_hidden(test_xs[test_task])
                    acc_tr   = knn_classifier(tr_hid,  tr_y,      feat, test_ys[test_task], num_classes=args.n_classes)[0]
                    acc_buf  = knn_classifier(bf_hid,  buffer.by, feat, test_ys[test_task], num_classes=args.n_classes)[0]
                    knns_tr += [acc_tr]
                    knns_buf += [acc_buf]
                    to_log[f'knn_not_buffer_{test_task}'] =  acc_tr
                    to_log[f'knn_buffer_{test_task}']     =  acc_buf

                print(i, 'train', [f' {x:.2f}' for x in knns_tr], f'{np.mean(knns_tr):.2f}', '\tbuffer', [f' {x:.2f}' for x in knns_buf], f'{np.mean(knns_buf):.2f}')

                model.train()


        if i % 20 == 0: print(f'{i} / {len(train_loader)}', end='\r')
        unique += y.unique().size(0)

        if n_seen > args.samples_per_task > 0: break

        if device == 'cuda':
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)

        inc_data = {'x': x, 'y': y, 't': task}

        agent.observe(inc_data)
        n_seen += x.size(0)

        if (task + i) > 100:
            # compute drift
            with torch.no_grad():
                post_tr_hid = F.normalize(model.return_hidden(tr_x), p=2, dim=-1)
                post_bf_hid = F.normalize(model.return_hidden(bf_x), p=2, dim=-1)
                to_log['drift_not_buffer'] = F.mse_loss(post_tr_hid, tr_hid).item()
                to_log['drift_buffer'] = F.mse_loss(post_bf_hid, bf_hid).item()

        wandb.log(to_log)
