# vim: expandtab:ts=4:sw=4
#
# Code to train on multiple video dataset [1].
#
#
# Example call to train a model using the cosine softmax classifier:
#
# ```
# python train_dji.py \
#     --dataset_dir=./dji \
#     --loss_mode=cosine-softmax \
#     --log_dir=./output/dji \
#     --run_id=cosine-softmax
# ```
#
# Example call to run evaluation on validation set (parallel to training):
#
# ```
# CUDA_VISIBLE_DEVICES="" python2 train_dji.py \
#     --dataset_dir=./dji \
#     --loss_mode=cosine-softmax \
#     --log_dir=./output/dji \
#     --run_id=cosine-softmax \
#     --mode=eval \
#     --eval_log_dir=./eval_output
# ```
#
# Example call to freeze a trained model (here model.ckpt-100000):
#
# ```
# python train_dji.py \
#     --restore_path=./output/dji/cosine-softmax/model.ckpt-100000 \
#     --mode=freeze
#
import functools
import os
import argparse
import tensorflow as tf
import numpy as np
#import scipy.io as sio
import train_app
from datasets import dji
from datasets import util
import nets.deep_sort.network_definition as net
#CUDA_VISIBLE_DEVICES = 0
#tf.config.experimental.set_memory_growth = True
tf.compat.v1.disable_v2_behavior()
#TF_FORCE_GPU_ALLOW_GROWTH=True
#config = tf.compat.v1.ConfigProto()
#config.gpu_options.allow_growth = True
#session = tf.compat.v1.InteractiveSession(config=config)

def get_max_label(data_path):
    ### image_train
    vid_list = []
    for (dirpath, dirnames, filenames) in os.walk(data_path):
        for filename in filenames:
            filename_base, ext = os.path.splitext(filename)
            vehicle_id, _, _,_ = filename_base.split('_')
            vid_list.append(vehicle_id)
    np.set_printoptions(edgeitems=50,linewidth=50)
    a = np.unique(vid_list)
    a = a.astype(np.int)
    max_label = np.amax(a, axis=0)
    return max_label


def create_default_argument_parser(dataset_name):
    """Create an argument parser with default arguments.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset. This value is used to set default directories.

    Returns
    -------
    argparse.ArgumentParser
        Returns an argument parser with default arguments.

    """
    parser = argparse.ArgumentParser(
        description="Metric trainer (%s)" % dataset_name)
    parser.add_argument(
        "--dataset_dir", help="Path to DJI dataset directory.",
        default="./resources/DJI/image_train")
    parser.add_argument(
        "--batch_size", help="Training batch size", default=128, type=int)
    parser.add_argument(
        "--learning_rate", help="Learning rate", default=1e-3, type=float)
    """parser.add_argument(
        "--eval_log_dir",
        help="Evaluation log directory (only used in mode 'evaluation').",
        default="./eval_log_dir/DJI")   # default="/tmp/%s_evaldir" % dataset_name)"""
    parser.add_argument(
        "--number_of_steps", help="Number of train/eval steps. If None given, "
        "runs infinitely", default=None, type=int)
    parser.add_argument(
        "--log_dir", help="Log and checkpoints directory.",
        default="./log/DJI")   #default="./log/%s_logdir" % dataset_name)
    parser.add_argument(
        "--loss_mode", help="One of 'cosine-softmax', 'magnet', 'triplet'",
        type=str, default="cosine-softmax")
    parser.add_argument(
        "--mode", help="One of 'train','freeze'.", #"One of 'train', 'eval', 'finalize', 'freeze'."
        type=str, default="train")
    parser.add_argument(
        "--restore_path", help="If not None, resume training of a given "
        "checkpoint (mode 'train').", default='./log/DJI/10616-ckpt/model.ckpt-10616')
    parser.add_argument(
        "--run_id", help="An optional run-id. If None given, a new one is "
        "created", type=str, default="cosine-softmax")
    return parser.parse_args()


class Data_Split(object):

    def __init__(self, dataset_dir, num_validation_y=0.1, seed=1234):
        self._dataset_dir = dataset_dir
        self._num_validation_y = num_validation_y
        self._seed = seed

    def read_train(self):
        images, ids, camera_indices = dji.read_image_directory_to_image(
            self._dataset_dir)
        train_indices, _ = util.create_validation_split(
            np.asarray(ids, np.int64), self._num_validation_y, self._seed)

        images = images[train_indices]
        ids = ids[train_indices]
        camera_indices = camera_indices[train_indices]
        return images, ids, camera_indices

    def read_validation(self):
        images, ids, camera_indices = dji.read_train_split_to_image(
            self._dataset_dir)
        _, valid_indices = util.create_validation_split(
            np.asarray(ids, np.int64), self._num_validation_y, self._seed)

        images = images[valid_indices]
        ids = ids[valid_indices]
        camera_indices = camera_indices[valid_indices]
        return images, ids, camera_indices


def main():
    args = create_default_argument_parser("dji")
    dataset = Data_Split(args.dataset_dir, num_validation_y=0.1, seed=1234)
    MAX_LABEL = get_max_label(args.dataset_dir)  # image_train 2172 image_test 2451
    if args.mode == "train":
        train_x, train_y, _ = dataset.read_train()
        print("Train set size: %d images, %d identites" % (
            len(train_x), len(np.unique(train_y))))   # 34267 images,  519  id

        network_factory = net.create_network_factory(
            is_training=True, num_classes=MAX_LABEL +1,
            add_logits=args.loss_mode == "cosine-softmax")
        train_kwargs = train_app.to_train_kwargs(args)
        train_app.train_loop(
            net.preprocess, network_factory, train_x, train_y,
            num_images_per_id=4, image_shape=dji.IMAGE_SHAPE,
            **train_kwargs)
    elif args.mode == "eval":
        valid_x, valid_y, camera_indices = dataset.read_validation()
        print("Validation set size: %d images, %d identites" % (
            len(valid_x), len(np.unique(valid_y))))

        network_factory = net.create_network_factory(
            is_training=False, num_classes=MAX_LABEL + 1,
            add_logits=args.loss_mode == "cosine-softmax")
        eval_kwargs = train_app.to_eval_kwargs(args)
        train_app.eval_loop(
            net.preprocess, network_factory, valid_x, valid_y, camera_indices,
            image_shape=dji.IMAGE_SHAPE, **eval_kwargs)
    elif args.mode == "export":
        raise NotImplementedError()
    elif args.mode == "finalize":
        network_factory = net.create_network_factory(
            is_training=False, num_classes=MAX_LABEL + 1,
            add_logits=False, reuse=None)
        train_app.finalize(
            functools.partial(net.preprocess, input_is_bgr=True),
            network_factory, args.restore_path, image_shape=dji.IMAGE_SHAPE,
            output_filename="./dji.ckpt")
    elif args.mode == "freeze":
        network_factory = net.create_network_factory(
            is_training=False, num_classes=MAX_LABEL + 1,
            add_logits=False, reuse=None)
        train_app.freeze(
            functools.partial(net.preprocess, input_is_bgr=True),
            network_factory, args.restore_path, image_shape=dji.IMAGE_SHAPE,
            output_filename="./freeze_dji-10616.pb")
    else:
        raise ValueError("Invalid mode argument.")


if __name__ == "__main__":
    CUDA_VISIBLE_DEVICES = ""
    #physical_devices = tf.compat.v2.config.list_physical_devices('GPU')
    #print("Num GPUs:", len(physical_devices))
    #print(tf.__version__)  2.4.1
    main()
