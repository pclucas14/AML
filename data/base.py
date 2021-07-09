import os
import sys
import pdb
import torch
import numpy as np

from copy import deepcopy
from torchvision import datasets, transforms
from data import *

# ---  Samplers --- #
class ContinualSampler(torch.utils.data.Sampler):

    def __init__(self, dataset, n_tasks):
        self.ds = dataset
        self.n_tasks = n_tasks
        self.classes = np.unique(dataset.targets)
        self.n_classes = self.classes.shape[0]

        assert self.n_classes % n_tasks == 0
        classes_per_task = self.cpt = self.n_classes // n_tasks

        self.task = None
        self.target_indices = {}

        for label in self.classes:
            self.target_indices[label] = \
                    np.squeeze(np.argwhere(self.ds.targets == label))


    def _fetch_task_samples(self, task):
        task = self.task
        task_classes = self.classes[self.cpt * task: self.cpt * (task + 1)]

        task_samples = []

        for task in task_classes:
            t_indices = self.target_indices[task]
            task_samples += [t_indices]

        task_samples = np.concatenate(task_samples)
        np.random.shuffle(task_samples)

        self.task_samples = task_samples


    def __iter__(self):
        self._fetch_task_samples(self.task)

        for item in self.task_samples:
            yield item


    def __len__(self):
        return len(self.task_samples)


    def set_task(self, task):
        self.task = task


def make_val_from_train(dataset, split=.9):
    train_ds, val_ds = deepcopy(dataset), deepcopy(dataset)

    train_idx, val_idx = [], []
    for label in np.unique(dataset.targets):
        label_idx = np.squeeze(np.argwhere(dataset.targets == label))
        split_idx = int(label_idx.shape[0] * split)
        train_idx += [label_idx[:split_idx]]
        val_idx   += [label_idx[split_idx:]]

    train_idx = np.concatenate(train_idx)
    val_idx   = np.concatenate(val_idx)

    train_ds.data = train_ds.data[train_idx]
    train_ds.targets = np.array(train_ds.targets)[train_idx]

    val_ds.data = val_ds.data[val_idx]
    val_ds.targets = np.array(val_ds.targets)[val_idx]

    return train_ds, val_ds


def get_data_and_tfs(args):
    dataset = {'split_cifar10' : CIFAR10,
               'split_cifar100': CIFAR100,
               'miniimagenet'  : MiniImagenet}[args.dataset]

    if args.n_tasks == -1:
        args.n_tasks = dataset.default_n_tasks

    if args.H is None:
        H = dataset.default_size
    else:
        H = args.H

    args.input_size = (3, H, H)

    base_tf  = dataset.base_transforms(H=H)
    train_tf = dataset.train_transforms(H=H, use_augs=args.use_augs)
    eval_tf  = dataset.eval_transforms(H=H)

    ds_kwargs = {'root': args.data_root, 'download': args.download}

    val_ds = test_ds = None
    if args.validation:
        trainval_ds      = dataset(train=True, **ds_kwargs)
        train_ds, val_ds = make_val_from_train(trainval_ds)
        train_ds.transform = base_tf
        val_ds.transform   = eval_tf
    else:
        train_ds         = dataset(train=True, transform=base_tf, **ds_kwargs)
        test_ds          = dataset(train=False, transform=eval_tf, **ds_kwargs)

    train_sampler = ContinualSampler(train_ds, args.n_tasks)
    train_loader  = torch.utils.data.DataLoader(
        train_ds,
        sampler=train_sampler,
        num_workers=args.n_workers,
        batch_size=args.batch_size,
    )

    if val_ds is not None:
        val_sampler = ContinualSampler(val_ds, args.n_tasks)
        test_loader = None
        val_loader  = torch.utils.data.DataLoader(
            val_ds,
            batch_size=128,
            sampler=val_sampler,
            num_workers=args.n_workers
        )

    elif test_ds is not None:
        test_sampler  = ContinualSampler(test_ds,  args.n_tasks)
        val_loader  = None
        test_loader = torch.utils.data.DataLoader(
            test_ds,
            batch_size=128,
            sampler=test_sampler,
            num_workers=args.n_workers
        )


    args.n_classes = train_sampler.n_classes

    return train_tf, train_loader, val_loader, test_loader
