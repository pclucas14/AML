import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

import sys
import itertools as it
import pandas as pd
from collections import OrderedDict

config = {
        '--dataset': ['miniimagenet', 'split_cifar10', 'split_cifar100'],
        '--mem_size' : ['20', '50', '100'],
        '--method': ['er', 'triplet', 'mask']
        }

cmd  = 'python clean_main.py --log online --reproc 0 --disc_iters 3'

n_runs = 20

log_dir = '/home/ml/users/lpagec/pytorch/mask-trick/results_hp1'
run_list_file = os.path.join(log_dir, 'run_list2.csv')
# check if config file has already been made

if not os.path.exists(run_list_file):
    # maks the run dist path
    args = sorted(config)
    combinations = it.product(*(config[arg] for arg in args))
    all_runs = list(combinations)
    all_runs_dict = []

    for run in all_runs:
        run_dict = {}
        for i, arg in enumerate(args):
            run_dict[arg] = run[i]

        run_dict['n_runs'] = n_runs
        all_runs_dict += [run_dict]

    out = pd.DataFrame(all_runs_dict)

    out.to_csv(run_list_file, index=False)


# --- actual running --- #
while True:
    # open the file
    run_list = pd.read_csv(run_list_file)
    if sum(run_list['n_runs']) <= 0:
        exit()

    # pick run with the most runs remaining
    run_id = run_list['n_runs'].argmax()
    run_values = OrderedDict(run_list.iloc[run_id])

    # execute said run
    run_cmd = cmd
    exp_name = ''

    for k, v in run_values.items():
        if k.startswith('--'):
            run_cmd += f' {k} {v}'
            exp_name += f'_{v}'

    # exp name
    run_cmd += f' --exp_name HP1{exp_name[1:]}_iters3'

    # execute command
    value = os.system(run_cmd)
    if value != 0:
        print('FAILURE')
        exit()

    # once done, decrease counter
    run_list = pd.read_csv(run_list_file)
    run_list.at[run_id, 'n_runs'] -= 1
    run_list.to_csv(run_list_file, index=False)

    print(run_list)


