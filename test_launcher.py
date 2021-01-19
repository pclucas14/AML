import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

import numpy as np
import sys
import itertools as it
import pandas as pd
from collections import OrderedDict

# cmd  = 'python clean_main.py --log online --reproc 0 --wandb_project er_imbalance_test'
cmd  = 'python clean_main.py --log online --reproc 0 --wandb_project er_imbalance_test --use_augmentations 1'

# all default runs
runs = [
    '--dataset miniimagenet --method er --mem_size 20 --lr 0.05',
    '--dataset miniimagenet --method er --mem_size 50 --lr 0.05',
    '--dataset miniimagenet --method er --mem_size 100 --lr 0.05',

    '--dataset miniimagenet --method icarl --mem_size 20 --icarl_gamma 0.9 --lr 0.1',
    '--dataset miniimagenet --method icarl --mem_size 50 --icarl_gamma 0.9 --lr 0.1',
    '--dataset miniimagenet --method icarl --mem_size 100 --icarl_gamma 0.9 --lr 0.1',

    '--dataset miniimagenet --method mask --mem_size 20 --lr 0.1',
    '--dataset miniimagenet --method mask --mem_size 50 --lr 0.1',
    '--dataset miniimagenet --method mask --mem_size 100 --lr 0.1',

    '--dataset miniimagenet --method triplet --mem_size 20 --margin 0.2 --buffer_neg 0',
    '--dataset miniimagenet --method triplet --mem_size 50 --margin 0.2 --buffer_neg 0',
    '--dataset miniimagenet --method triplet --mem_size 100 --margin 0.2 --buffer_neg 0',


    '--dataset split_cifar10 --method er --mem_size 20 --lr 0.05',
    '--dataset split_cifar10 --method er --mem_size 50 --lr 0.1',
    '--dataset split_cifar10 --method er --mem_size 100 --lr 0.1',

    '--dataset split_cifar10 --method icarl --mem_size 20 --icarl_gamma 0.9 --lr 0.05',
    '--dataset split_cifar10 --method icarl --mem_size 50 --icarl_gamma 0.5 --lr 0.05',
    '--dataset split_cifar10 --method icarl --mem_size 100 --icarl_gamma 0.9 --lr 0.1',

    '--dataset split_cifar10 --method mask --mem_size 20 --lr 0.1',
    '--dataset split_cifar10 --method mask --mem_size 50 --lr 0.1',
    '--dataset split_cifar10 --method mask --mem_size 100 --lr 0.1',

    '--dataset split_cifar10 --method triplet --mem_size 20 --margin 0.2 --buffer_neg 0 --lr 0.05',
    '--dataset split_cifar10 --method triplet --mem_size 50 --margin 0.2 --buffer_neg 0 --lr 0.05',
    '--dataset split_cifar10 --method triplet --mem_size 100 --margin 0.2 --buffer_neg 0 --lr 0.05',


    '--dataset split_cifar100 --method er --mem_size 20 --lr 0.1',
    '--dataset split_cifar100 --method er --mem_size 50 --lr 0.1',
    '--dataset split_cifar100 --method er --mem_size 100 --lr 0.1',

    '--dataset split_cifar100 --method icarl --mem_size 20 --icarl_gamma 0.9 --lr 0.1',
    '--dataset split_cifar100 --method icarl --mem_size 50 --icarl_gamma 0.9 --lr 0.1',
    '--dataset split_cifar100 --method icarl --mem_size 100 --icarl_gamma 0.9 --lr 0.1',

    '--dataset split_cifar100 --method mask --mem_size 20 --lr 0.1',
    '--dataset split_cifar100 --method mask --mem_size 50 --lr 0.1',
    '--dataset split_cifar100 --method mask --mem_size 100 --lr 0.1',

    '--dataset split_cifar100 --method triplet --mem_size 20 --margin 0.2 --buffer_neg 0 --lr 0.05',
    '--dataset split_cifar100 --method triplet --mem_size 50 --margin 0.2 --buffer_neg 0 --lr 0.05',
    '--dataset split_cifar100 --method triplet --mem_size 100 --margin 0.2 --buffer_neg 0 --lr 0.05',
]

# iid baseline runs
runs = [
    '--dataset miniimagenet --method iid',
    '--dataset miniimagenet --method iid++',

    '--dataset split_cifar10 --method iid',
    '--dataset split_cifar10 --method iid++',

    '--dataset split_cifar100 --method iid',
    '--dataset split_cifar100 --method iid++',
]

# with augmentation
runs = [
    '--dataset miniimagenet --method mask --mem_size 20 --lr 0.1',
    '--dataset miniimagenet --method mask --mem_size 50 --lr 0.1',
    '--dataset miniimagenet --method mask --mem_size 100 --lr 0.1',

    '--dataset miniimagenet --method triplet --mem_size 20 --margin 0.2 --buffer_neg 0',
    '--dataset miniimagenet --method triplet --mem_size 50 --margin 0.2 --buffer_neg 0',
    '--dataset miniimagenet --method triplet --mem_size 100 --margin 0.2 --buffer_neg 0',

    '--dataset split_cifar10 --method mask --mem_size 20 --lr 0.1',
    '--dataset split_cifar10 --method mask --mem_size 50 --lr 0.1',
    '--dataset split_cifar10 --method mask --mem_size 100 --lr 0.1',

    '--dataset split_cifar10 --method triplet --mem_size 20 --margin 0.2 --buffer_neg 0 --lr 0.05',
    '--dataset split_cifar10 --method triplet --mem_size 50 --margin 0.2 --buffer_neg 0 --lr 0.05',
    '--dataset split_cifar10 --method triplet --mem_size 100 --margin 0.2 --buffer_neg 0 --lr 0.05',

    '--dataset split_cifar100 --method mask --mem_size 20 --lr 0.1',
    '--dataset split_cifar100 --method mask --mem_size 50 --lr 0.1',
    '--dataset split_cifar100 --method mask --mem_size 100 --lr 0.1',

    '--dataset split_cifar100 --method triplet --mem_size 20 --margin 0.2 --buffer_neg 0 --lr 0.05',
    '--dataset split_cifar100 --method triplet --mem_size 50 --margin 0.2 --buffer_neg 0 --lr 0.05',
    '--dataset split_cifar100 --method triplet --mem_size 100 --margin 0.2 --buffer_neg 0 --lr 0.05',
]



n_runs = 20

log_dir = '/home/ml/users/lpagec/pytorch/mask-trick/results_test'

if not os.path.isdir(log_dir):
    print(f'making {log_dir} path')
    os.makedirs(log_dir)

run_list_file = os.path.join(log_dir, 'run_list_augs.csv')
# check if config file has already been made

if not os.path.exists(run_list_file):
    # maks the run dist path
    run_list = []

    for run in runs:
        exp_name = ''.join([x for x in run.split(' ') if not x.startswith('--')]) + 'use_aug'
        run_list += [f'{cmd} {run} --exp_name {exp_name}']

    runs = np.array(run_list)
    runs = np.stack((runs, np.array([f'{n_runs}' for _ in range(len(runs))])), axis=1)
    out = pd.DataFrame(data=runs, columns=['command', 'n_runs'])
    print('final list of runs ', out)

    out.to_csv(run_list_file, index=False)


# --- actual running --- #
while True:
    # open the file
    run_list = pd.read_csv(run_list_file)
    if sum(run_list['n_runs']) <= 0:
        exit()

    # pick run with the most runs remaining
    probs = run_list['n_runs'].values == run_list['n_runs'].max()
    probs = probs.astype('float32')
    probs = probs / probs.sum()

    run_id = np.random.choice(np.arange(probs.shape[0]), p=probs)
    # run_id = run_list['n_runs'].argmax()
    run_values = OrderedDict(run_list.iloc[run_id])

    # execute said run
    run_cmd = run_values['command']

    # execute command
    print(f'running {run_cmd}')
    value = os.system(run_cmd)
    if value != 0:
        print('FAILURE')
        exit()

    # once done, decrease counter
    run_list = pd.read_csv(run_list_file)
    run_list.at[run_id, 'n_runs'] -= 1
    run_list.to_csv(run_list_file, index=False)

    print(run_list)
