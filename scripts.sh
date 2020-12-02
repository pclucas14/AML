#python er_main.py --mem_size 20  --n_runs 1
#python er_main.py  --mem_size 20  --disc_iters 3 --n_runs 1
#python er_main.py --suffix 'baseline' --dataset split_cifar100 --log online --mem_size 20  --disc_iters 1 --n_runs 1 --buffer_batch_size 10
#python er_main.py --suffix 'baseline_mask' --dataset split_cifar100 --log online --mem_size 20  --mask_trick --mask_vers 1 --disc_iters 1 --n_runs 1 --buffer_batch_size 10


#python er_modular.py --suffix 'baseline_mask' --mask_trick  --log online --mem_size 20 --n_runs 1 --dataset miniimagenet --disc_iters 3 --n_runs 1 --buffer_batch_size 10

#python er_modular.py --suffix 'baseline_mask' --dataset miniimagenet --log online --mem_size 100  --mask_trick --mask_vers 4 --disc_iters 3 --n_runs 1 --buffer_batch_size 10


#python er_modular.py --suffix 'baseline_mask' --dataset split_cifar10 --log online --mem_size 20  --top_only --mask_trick --mask_vers 4 --disc_iters 1 --n_runs 1 --buffer_batch_size 10

#python er_modular.py --suffix 'baseline_mask_depth' --dataset split_cifar10 --log online --mem_size 20  --mask_trick --disc_iters 1 --n_runs 5 --buffer_batch_size 5

#python er_modular.py --suffix 'baseline_mask_M100' --dataset split_cifar10 --log online --mem_size 100  --top_only --mask_trick --mask_vers 4 --disc_iters 3 --n_runs 5 --buffer_batch_size 10

#python er_modular.py --suffix 'baseline_mask_depth_M100' --dataset split_cifar10 --log online --mem_size 100  --mask_trick --disc_iters 3 --n_runs 5 --buffer_batch_size 5



python er_main_contrastive.py --suffix 'baseline_mask' --lr 0.1 --dataset split_cifar10 --log online --mem_size 100  --mask_trick --disc_iters 3 --n_runs 1 --batch_size 10 --buffer_batch_size 10


