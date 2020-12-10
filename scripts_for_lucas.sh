
python er_main_contrastive_clean_for_lucas.py --suffix 'baseline_mask' --lr 0.1 --dataset split_cifar10  --buffer_neg 2.0 --incoming_neg 0.0  --mem_size 20   --disc_iters 1 --n_runs 1 --batch_size 10 --buffer_batch_size 10
python er_main_contrastive_clean_for_lucas.py --suffix 'baseline_mask' --lr 0.1 --dataset split_cifar10  --buffer_neg 0.0 --incoming_neg 2.0  --mem_size 20   --disc_iters 1 --n_runs 1 --batch_size 10 --buffer_batch_size 10




