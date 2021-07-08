## Reducing Representation Drift in Online Continual Learning

[![Paper](https://img.shields.io/badge/arXiv-2104.05025-brightgreen)](https://arxiv.org/abs/2104.05025)

Avoiding disrupting learned representations when new classes are introduced </br>

## (key) Requirements 
- Python 3.8
- Pytorch 1.6.0

## Structure

    ├── method.py           # Implementation of proposed methods and reported baselines
    ├── buffer.py           # Basic buffer implementation 
    ├── data.py             # DataLoaders and Datasets
    ├── main.py             # main file for ER
    ├── model.py            # model defitinition and cosine cross-entropy head
    ├── losses.py           # custom loss functions
    ├── utils.py            # utilities, from logging to data download

## Running Experiments

<!-- * er = Experience Replay baseline
* mask = ER-ACE
* supcon = ER-AML
* triplet = ER-AML Triplet -->

```
python main.py --dataset <dataset> --method <method> --mem_size <mem_size> 
```
ER-ACE example:

```
python main.py --dataset split_cifar10 --method er_ace --mem_size 20 
```
ER-AML example:

```
python main.py --dataset split_cifar10 --method er_aml --mem_size 20 --supcon_temperature 0.2
```

## Logging

Logging is done with [Weights & Biases](https://www.wandb.com/) and can be turned on like this: </br>
```
python -dataset <dataset> --method <method> --wandb_log online --wandb_entity <wandb username>
```

## Cite
```
@article{caccia2021reducing,
  title={Reducing Representation Drift in Online Continual Learning},
  author={Caccia, Lucas and Aljundi, Rahaf and Tuytelaars, Tinne and Pineau, Joelle and Belilovsky, Eugene},
  journal={arXiv preprint arXiv:2104.05025},
  year={2021}
}
```