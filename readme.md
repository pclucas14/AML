## Reducing Drift in Online Representation Learning

Avoiding disrupting learned representations when new classes are introduced </br>

## (key) Requirements 
- Python 3.8
- Pytorch 1.6.0

## Structure

    ├── method.py           # Implementation of proposed methods and reported baselines
    ├── buffer.py           # Basic buffer implementation 
    ├── data.py             # DataLoaders and Datasets
    ├── main.py             # main file for ER
    ├── utils.py            # utilities, from logging to data download

## Running Experiments

```
python main.py --dataset <dataset> --method <method> --mem_size <mem_size> 
```
e.g.

```
python main.py --dataset split_cifar10 --method mask --mem_size 20 
```


