#python er_main.py --mem_size 20  --n_runs 1
#python er_main.py  --mem_size 20  --disc_iters 3 --n_runs 1
#python er_main.py --suffix 'baseline' --dataset split_cifar100 --log online --mem_size 20  --disc_iters 1 --n_runs 1 --buffer_batch_size 10
#python er_main.py --suffix 'baseline_mask' --dataset split_cifar100 --log online --mem_size 20  --mask_trick --mask_vers 1 --disc_iters 1 --n_runs 1 --buffer_batch_size 10


python er_modular.py --suffix 'baseline_mask' --dataset split_cifar10 --log online --mem_size 20  --mask_trick --mask_vers 4 --disc_iters 1 --n_runs 1 --buffer_batch_size 10



#python er_main.py --suffix 'baseline_mask' --dataset split_cifar10 --log online --mem_size 20  --mask_trick --mask_vers 5 --disc_iters 1 --n_runs 1 --buffer_batch_size 10


