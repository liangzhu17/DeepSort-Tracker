# cosine_metric_learning

## Introduction

This repository contains code for training a metric feature representation to be
used with the [deep_sort tracker](https://github.com/nwojke/deep_sort). The
approach is described in

    @inproceedings{Wojke2018deep,
      title={Deep Cosine Metric Learning for Person Re-identification},
      author={Wojke, Nicolai and Bewley, Alex},
      booktitle={2018 IEEE Winter Conference on Applications of Computer Vision (WACV)},
      year={2018},
      pages={748--756},
      organization={IEEE},
      doi={10.1109/WACV.2018.00087}
    }

## Training 
number_of_steps = None: infinite training

```
CUDA_VISIBLE_DEVICES="" python train_dji.py \
    --mode=train \
    --dataset_dir=./resources \
    --loss_mode=cosine-softmax \
    --log_dir=./log/DJI \
    --run_id=cosine-softmax \

```
The results of training process are stored in tensorboard:
```
tensorboard --logdir ./log/DJI/cosine-softmax --port 6006
```


## Model export

To export your trained model for use with the
[deep_sort tracker](https://github.com/nwojke/deep_sort), run the following
command:
```
python train_dji.py --mode=freeze --restore_path=PATH_TO_CHECKPOINT
```
This will create a ``mars.pb`` file which can be supplied to Deep SORT. Again,
the Market1501 script contains a similar function.
