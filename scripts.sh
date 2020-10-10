#python er_main.py --mem_size 20  --n_runs 1
#python er_main.py  --mem_size 20  --disc_iters 3 --n_runs 1
python er_main.py --dataset split_cifar10 --mem_size 20  --mask_trick --disc_iters 1 --n_runs 3 --buffer_batch_size 10

