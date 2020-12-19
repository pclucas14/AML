python clean_main.py --suffix 'er_augment' --log online --lr 0.1 --dataset split_cifar10 --buffer_neg 0.0 --method 'er' --augment --mem_size 20 --disc_iters 1 --n_runs 1  --batch_size 10 --buffer_batch_size 10
python clean_main.py --suffix 'er_triplet' --log online --lr 0.1 --dataset split_cifar10 --buffer_neg 0.0 --method 'triplet' --augment --mem_size 20 --disc_iters 1 --n_runs 1  --batch_size 10 --buffer_batch_size 10
python clean_main.py --suffix 'er_triplet_neg' --log online --lr 0.1 --dataset split_cifar10 --buffer_neg 2.0 --method 'triplet' --augment --mem_size 20 --disc_iters 1 --n_runs 1  --batch_size 10 --buffer_batch_size 10

#python clean_main.py --suffix 'woof' --lr 0.1 --dataset split_cifar10  --buffer_neg 0.0 --method 'triplet'  --mem_size 20  --disc_iters 1 --n_runs 1 --batch_size 10 --buffer_batch_size 10



