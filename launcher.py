import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

import numpy as np
import sys
import itertools as it
import pandas as pd
from collections import OrderedDict

config = {
        '--dataset': ['split_cifar10', 'split_cifar100', 'miniimagenet'],
        '--mem_size' : ['20', '50', '100'],
        '--method': ['der'], #'er', 'icarl', 'mask', 'triplet'],
        '--alpha': [0, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2],
        '--beta': [0, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2],
        '--use_augmentations': [1],
        '--lr': [0.05, 0.01, 0.1],
        }

# make sure we don't run useless configs
conditions = [
        lambda x : (x['--alpha'] == + x['--beta']) > 0 ,
]

check_conditions = lambda args, conds: \
        all([cond(args) for cond in conds])

cmd  = 'python main.py --wandb_log online --wandb_project er_rebuttal'

n_runs = 5

log_dir = '/home/ml/users/lpagec/pytorch/mask-trick/results_der'

if not os.path.isdir(log_dir):
    print(f'making {log_dir} path')
    os.makedirs(log_dir)

run_list_file = os.path.join(log_dir, 'run_der_ok_fixed_.csv')
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

        # make sure only valid configs are taken
        if not check_conditions(run_dict, conditions):
            continue

        run_dict['n_runs'] = n_runs
        all_runs_dict += [run_dict]

    out = pd.DataFrame(all_runs_dict)

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
    run_cmd = cmd
    exp_name = ''

    for k, v in run_values.items():
        if k.startswith('--'):
            run_cmd += f' {k} {v}'
            exp_name += f'_{k[2:10]}:{v}'

    # exp name
    run_cmd += f' --exp_name {exp_name[1:]}'

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
