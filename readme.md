### Results

```
python er_main.py --mem_size 10 --mask_trick --n_runs 20

Final valid Accuracy: 0.356 +/- 0.009
Final valid Forget: 0.310 +/- 0.015
```


```
python er_main.py --mem_size 20 --mask_trick --n_runs 20

Final valid Accuracy: 0.394 +/- 0.010
Final valid Forget: 0.270 +/- 0.012
```


```
python er_main.py --mem_size 50 --mask_trick --n_runs 20

Final valid Accuracy: 0.428 +/- 0.009
Final valid Forget: 0.221 +/- 0.017
```


```
python er_main.py --mem_size 100 --mask_trick --n_runs 20

Final valid Accuracy: 0.470 +/- 0.012
Final valid Forget: 0.172 +/- 0.018
```

```
python er_main.py --mem_size 20 --n_runs 20 --dataset miniimagenet --disc_iters 3

Final valid Accuracy: 0.103 +/- 0.004
Final valid Forget: 0.481 +/- 0.006


python er_main.py --mem_size 20 --mask_trick --n_runs 20 --dataset miniimagenet --disc_iters 3

Final valid Accuracy: 0.183 +/- 0.003
Final valid Forget: 0.171 +/- 0.006

```


### Notes
1. concatenating incoming and rehearsal batches really degrades performance here (at least for M=20)
2. what about using `exclude_task` ? It you *do* exclude tasks, the model performs really bad on the incoming tasks. Like very bad. Maybe this suggests that the only learning occuring is on the buffered samples. 
3. If we train *only* on the buffer (except for 1st task), model does really bad on all tasks, and okay-ish on last one. 
3. batch size = `5` seems to be almost as good
4. If we use the exact same masks, but we concatenate streaming and rehearsal inputs, the performance gain vanishes (`23% acc, M=20`) with the mask trick. Maybe this hints towards something batch-norm related. 
5. if we interpret the setup as an extreme class imbalance scenario, could we track an EMA of last seen classes and weight samples according to the inverse frequencies ?

### Logit norms
```
masking 
[0.8407, 0.8302, 0.3043, 0.2934, 0.3309, 0.3518, 0.3428, 0.3303, 0.3071, 0.3836]
[4.3777, 4.6886, 1.7205, 1.5952, 0.3309, 0.3518, 0.3428, 0.3303, 0.3071, 0.3836]
[6.6236, 6.7833, 4.5002, 4.2572, 1.4884, 1.4059, 0.3428, 0.3303, 0.3071, 0.3836]
[7.6814, 7.8651, 5.9000, 5.4516, 3.6023, 3.0890, 1.6171, 1.8452, 0.3071, 0.3836]
[7.9753, 8.5349, 6.4735, 6.4533, 4.8286, 3.8534, 3.0717, 3.3976, 1.7224, 1.8973]
```

```
no masking
[2.4916, 2.7374, 0.4281, 0.4157, 0.4535, 0.4757, 0.4706, 0.4680, 0.4372, 0.5047]
[6.6366, 8.6012, 5.0396, 5.6869, 0.6778, 0.6966, 0.6946, 0.7002, 0.6656, 0.7178]
[9.9267, 11.699, 9.1237, 8.8783, 4.6636, 5.2733, 0.8565, 0.8690, 0.8337, 0.8776]
[12.087, 13.855, 11.940, 11.702, 8.111,  8.0370, 4.0227, 4.1022, 0.9274, 0.9774]
[13.667, 15.364, 13.475, 13.552, 10.466, 9.6046, 7.2251, 6.7900, 4.1522, 4.0249]
```


### Other
```
python er_main.py --mem_size 100 --mask_trick --n_runs 5 --n_tasks 1
Final valid Accuracy: 0.638 +/- 0.010

python er_main.py --mem_size 0 --mask_trick --n_runs 5 --n_tasks 1
Final valid Accuracy: 0.578 +/- 0.029

python er_main.py --mem_size 500  --n_runs 5 --n_tasks 1 --task_free
Final valid Accuracy: 0.710 +/- 0.006

python er_main.py --mem_size 500 --mask_trick --n_runs 5
Final valid Accuracy: 0.560 +/- 0.020
Final valid Forget: 0.112 +/- 0.026

python er_main.py --mem_size 500  --n_runs 5
(no excluding tasks) : 
Final valid Accuracy: 0.500 +/- 0.042
Final valid Forget: 0.376 +/- 0.049

(excluding tasks)
Final valid Accuracy: 0.460 +/- 0.034
Final valid Forget: 0.146 +/- 0.034


python er_main.py --mem_size 100 --n_runs 5 --dataset miniimagenet --n_tasks 1 --task_free --disc_iters 1
Final valid Accuracy: 0.242 +/- 0.010

python er_main.py --mem_size 100 --n_runs 5 --dataset miniimagenet --n_tasks 1 --task_free --disc_iters 3
Final valid Accuracy: 0.280 +/- 0.013

python er_main.py --mem_size 100 --n_runs 5 --dataset miniimagenet --task_free --disc_iters
Final valid Accuracy: 0.246 +/- 0.018
Final valid Forget: 0.324 +/- 0.021

python er_main.py --mem_size 100 --n_runs 5 --dataset miniimagenet --task_free --disc_iters 3 --mask_trick
Final valid Accuracy: 0.238 +/- 0.010
Final valid Forget: 0.104 +/- 0.012

python er_main.py --mem_size 500 --n_runs 20 --dataset miniimagenet --task_free --disc_iters 3
Final valid Accuracy: 0.288 +/- 0.010
Final valid Forget: 0.297 +/- 0.009

python er_main.py --mem_size 500 --n_runs 20 --dataset miniimagenet --task_free --disc_iters 3 --mask_trick
Final valid Accuracy: 0.268 +/- 0.007
Final valid Forget: 0.097 +/- 0.006
```
