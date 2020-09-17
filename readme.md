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

### Notes
1. concatenating incoming and rehearsal batches really degrades performance here (at least for M=20)
2. what about using `exclude_task` ? It you *do* exclude tasks, the model performs really bad on the incoming tasks. Like very bad. Maybe this suggests that the only learning occuring is on the buffered samples. 
3. If we train *only* on the buffer (except for 1st task), model does really bad on all tasks, and okay-ish on last one. 
3. batch size = `5` seems to be almost as good
x. if we interpret the setup as an extreme class imbalance scenario, could we track an EMA of last seen classes and weight samples according to the inverse frequencies ?
