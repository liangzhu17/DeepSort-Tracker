# Deep SORT

## Introduction

This repository contains code for *Simple Online and Realtime Tracking with a Deep Association Metric* (Deep SORT).
It also contains code from *HOTA (and other) evaluation metrics for Multi-Object Tracking (MOT)*.

## Dependencies 

The code is compatible with Python 2.7 and 3. The following dependencies are
needed to run the tracker:

* NumPy
* sklearn
* OpenCV

Additionally, feature generation requires TensorFlow (>= 1.0).

## Installation
First, clone the repository:
```
git clone https://gitlab.itd-services.de/trafficobservation/research/deepsort.git
```
Then, solve the environments and dependencies
```
pip install -r requirements.txt 
```
## Data preparation

* **DJI-Videos and csv files**
You should prepare the data in the following structure:
The csv file which is derived from its video file should have the same number in their filenames.
The string parts should ste

```
deepsort_tracker
   |——————dataset
   |        └——————vid_input 
   |        |            └——————string1_number1.mp4----(eg. DJI_08.mp4)
   |        |            └——————string1_number2.mp4----(eg. DJI_12.mp4)
   |	    | 		 └——————...
   |	    |
   |        └——————csv_input (yolo bounding boxes)
   |        |            └——————string2_number1.mp4----(eg. CSV_08.csv)
   |        |            └——————string2_number2.mp4----(eg. CSV_12.csv)
   |        |            └——————...
   |        | 
   |        └——————data_crop_prep
   |                     └——————train(empty) ---- directory for cropped images 
   |                                     └——————-xxx.jpg 
   |                                     └——————-... 
   |
   |——————model_data
   |        └——————freeze_dji-10616.pb(default reid model after training)
   .....
```
You can also change the input data directories in tracker_app.py function create_tracker_arg_parser()
and in Crop_bbox_for_train.py function create_crop_arg_parser()

## Running the tracker

The following example starts the tracker on multiple mp4 videos.
We assume resources have been put to the repository root directory seperately in video_dir_in and csv_dir_in:

```
python tracker_app.py \
    --video_dir_in=./dataset/vid_input \
    --csv_dir_in=./dataset/csv_input \
    --video_dir_out=./tracker_output \
    --model_path=./model_data/freeze_dji-10616.pb
```

## Prepare training data

Run follows to create cropped image files from original videos with bounding box annotations.
 
```
python Crop_bbox_for_train.py \
    --video_dir_in=./dataset/vid_input \
    --csv_dir_in=./dataset/csv_input \
    --crop_dir_out=./dataset/data_crop_prep_try/train
```

## Training the model

To train the deep association metric model we used a novel [cosine metric learning](https://github.com/nwojke/cosine_metric_learning) approach which is provided as a separate repository cosine_metric_learning.

## Highlevel overview of source files

In the top-level directory are executable scripts to execute, evaluate, and
visualize the tracker. The main entry point is in `tracker_app.py`.
This file runs the tracker on a MOTChallenge sequence.

In package `deep_sort` is the main tracking code:

* `detection.py`: Detection base class.
* `kalman_filter.py`: A Kalman filter implementation and concrete
   parametrization for image space filtering.
* `linear_assignment.py`: This module contains code for min cost matching and
   the matching cascade.
* `iou_matching.py`: This module contains the IOU matching metric.
* `nn_matching.py`: A module for a nearest neighbor matching metric.
* `track.py`: The track class contains single-target track data such as Kalman
  state, number of hits, misses, hit streak, associated feature vectors, etc.
* `tracker.py`: This is the multi-target tracker class.


## Citing DeepSORT

If you find this repo useful in your research, please consider citing the following papers:

    @inproceedings{Wojke2017simple,
      title={Simple Online and Realtime Tracking with a Deep Association Metric},
      author={Wojke, Nicolai and Bewley, Alex and Paulus, Dietrich},
      booktitle={2017 IEEE International Conference on Image Processing (ICIP)},
      year={2017},
      pages={3645--3649},
      organization={IEEE},
      doi={10.1109/ICIP.2017.8296962}
    }

    @inproceedings{Wojke2018deep,
      title={Deep Cosine Metric Learning for Person Re-identification},
      author={Wojke, Nicolai and Bewley, Alex},
      booktitle={2018 IEEE Winter Conference on Applications of Computer Vision (WACV)},
      year={2018},
      pages={748--756},
      organization={IEEE},
      doi={10.1109/WACV.2018.00087}
    }
