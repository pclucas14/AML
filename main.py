import time
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict as OD

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

agent = METHODS[args.method](model, logger, train_tf, args)
n_params = sum(np.prod(p.size()) for p in model.parameters())
buffer = agent.buffer

print(model)
print("number of classifier parameters:", n_params)

eval_accs = []
if args.validation:
    mode = 'valid'
    eval_loader = val_loader
else:
    mode = 'test'
    eval_loader = test_loader


# Get The Test Set from the first task
train_xs, train_ys = [], []
test_xs, test_ys = [], []
for i in range(args.n_tasks):
    train_loader.sampler.set_task(i)
    val_loader.sampler.set_task(i)
    aa, bb = next(iter(val_loader))
    test_xs += [aa.to(device)]
    test_ys += [bb.to(device)]

    for i, (aa, bb) in enumerate(train_loader):
        train_xs += [aa.to(device)]
        train_ys += [bb.to(device)]
        if i == 19: break

train_xs = torch.cat(train_xs)
train_ys = torch.cat(train_ys)

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

        if task>0 and (i<11 or i%10==0):
            with torch.no_grad():
                model.eval()
                knns_tr, knns_buf = [], []
                train_feats_buf = model.return_hidden(buffer.bx)
                train_feats_tr = model.return_hidden(train_xs)
                for test_task in range(args.n_tasks):
                    feat = model.return_hidden(test_xs[test_task])
                    acc_tr  = knn_classifier(train_feats_tr, train_ys, feat, test_ys[test_task], num_classes=args.n_classes)[0]
                    acc_buf  = knn_classifier(train_feats_buf, buffer.by, feat, test_ys[test_task], num_classes=args.n_classes)[0]
                    knns_tr += [acc_tr]
                    knns_buf += [acc_buf]

                print(i, 'train', [f' {x:.2f}' for x in knns_tr], f'{np.mean(knns_tr):.2f}', '\tbuffer', [f' {x:.2f}' for x in knns_buf], f'{np.mean(knns_buf):.2f}')

            model.train()


        if i % 20 == 0: print(f'{i} / {len(train_loader)}', end='\r')
        unique += y.unique().size(0)

        if n_seen > args.samples_per_task > 0: break

        if device == 'cuda':
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)

        inc_data = {'x': x, 'y': y, 't': task}


        '''
        if task > 0 and i<10:#(i < 5 or i % 150 == 0):
            model.eval()
            test_set_align, buffer_align = [], []
            for test_task in range(task + 1):
                idx = agent.buffer.bt == test_task
                if idx.int().sum() > 0:
                    test_set_align += [ measure_drift(model, test_xs[test_task], test_ys[test_task]).item() ]
                    buffer_align   += [ measure_drift(model, agent.buffer.bx[idx], agent.buffer.by[idx]).item() ]

            #print(f'i {i} : test set alignment\t', [f'{x:.2f}' for x in test_set_align])
            idx = torch.arange(args.n_classes_per_task) + task * args.n_classes_per_task
            #print(f'i {i} : Buffer alignment\t', [f'{x:.2f}' for x in buffer_align])
        '''
        '''
        test_set_align, buffer_align = [], []
        for test_task in range(task + 1):
            idx = agent.buffer.bt == test_task
            if idx.int().sum() > 0:
                test_set_align += [ measure_drift(model, test_xs[test_task], test_ys[test_task], proto=last_proto).item() ]
                buffer_align   += [ measure_drift(model, agent.buffer.bx[idx], agent.buffer.by[idx], proto=last_proto).item() ]

        print(f'i {i} : last test set alignment\t', [f'{x:.2f}' for x in test_set_align])
        idx = torch.arange(args.n_classes_per_task) + task * args.n_classes_per_task
        print(f'i {i} : last Buffer alignment\t', [f'{x:.2f}' for x in buffer_align])

        '''

        agent.observe(inc_data)
        n_seen += x.size(0)
'''
    # always eval at end of tasks
    print(f'Task {task} over. took {time.time() - start:.2f} Avg unique label {unique / i}')
    eval_accs  += [eval_agent(agent, eval_loader, task, mode=mode)]

    print('Task 0 test set drift\t', measure_drift(model, test_xs[0], test_ys[0]).item())

    idx = agent.buffer.by < args.n_classes_per_task
    print('Task 0 Buffer drift\t',  measure_drift(model, agent.buffer.bx[idx], agent.buffer.by[idx]).item())




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
}, verbose=True)

logger.close()
'''
with torch.no_grad():
    model.eval()
    knns = []
    train_feats = model.return_hidden(buffer.bx)#train_xs)
    for test_task in range(args.n_tasks):
        feat = model.return_hidden(test_xs[test_task])
        #acc  = knn_classifier(train_feats, train_ys, feat, test_ys[test_task], num_classes=args.n_classes)[0]
        acc  = knn_classifier(train_feats, buffer.by, feat, test_ys[test_task], num_classes=args.n_classes)[0]
        knns += [acc]

    print([f'{x:.2f}' for x in knns], np.mean(knns))

model.train()

