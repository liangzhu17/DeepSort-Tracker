
""" run_eval.py
###  python scripts/run_eval.py --GT_FOLDER '/input/gt' --OUTPUT_FOLDER '/workspace' --TRACKERS_FOLDER '/input/trackers'

Run example:
run_eval.py --USE_PARALLEL False --METRICS Hota --TRACKERS_TO_EVAL Lif_T
  #--csv_path_in /input/dataset/tracker_input/csv/01_Track.csv --video_dir_out /workspace --gt_eval /workspace --tracker_eval /workspace --model_path /input/model_data/freeze_dji-10616.pb
Command Line Arguments: Defaults, # Comments
    Eval arguments:
        'USE_PARALLEL': False,
        'NUM_PARALLEL_CORES': 8,
        'BREAK_ON_ERROR': True,
        'PRINT_RESULTS': True,
        'PRINT_ONLY_COMBINED': False,
        'PRINT_CONFIG': True,
        'TIME_PROGRESS': True,
        'OUTPUT_SUMMARY': True,
        'OUTPUT_DETAILED': True,
        'PLOT_CURVES': True,
    Dataset arguments:
        'GT_FOLDER': os.path.join(code_path, 'data/gt/mot_challenge/'),  # Location of GT data
        'TRACKERS_FOLDER': os.path.join(code_path, 'data/trackers/mot_challenge/'),  # Trackers location
        'OUTPUT_FOLDER': None,  # Where to save eval results (if None, same as TRACKERS_FOLDER)
        'TRACKERS_TO_EVAL': None,  # Filenames of trackers to eval (if None, all in folder)
        'CLASSES_TO_EVAL': ['pedestrian'],  # Valid: ['pedestrian']
        'BENCHMARK': 'MOT17',  # Valid: 'MOT17', 'MOT16', 'MOT20', 'MOT15'
        'SPLIT_TO_EVAL': 'train',  # Valid: 'train', 'test', 'all'
        'INPUT_AS_ZIP': False,  # Whether tracker input files are zipped
        'PRINT_CONFIG': True,  # Whether to print current config
        'DO_PREPROC': True,  # Whether to perform preprocessing (never done for 2D_MOT_2015)
        'TRACKER_SUB_FOLDER': 'data',  # Tracker files are in TRACKER_FOLDER/tracker_name/TRACKER_SUB_FOLDER
        'OUTPUT_SUB_FOLDER': '',  # Output files are saved in OUTPUT_FOLDER/tracker_name/OUTPUT_SUB_FOLDER
    Metric arguments:
        'METRICS': ['HOTA', 'CLEAR', 'Identity', 'VACE']
"""

import sys
import os
import argparse
from multiprocessing import freeze_support
import numpy
import tensorflow as tf
import os.path as osp
# CUDA_VISIBLE_DEVICES = 0
# tf.config.experimental.set_memory_growth = True
tf.compat.v1.disable_v2_behavior()


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import trackeval  # noqa: E402


def mkdirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def create_eval_arg_parser():
    """Parse command line arguments.
        """
    arg_parser = argparse.ArgumentParser(
        description="Evaluation process")
    arg_parser.add_argument(
        "--gt_fol", help="Directory to ground truth input file",
        default='./eval_data/gt')
    arg_parser.add_argument(
        "--out_fol", help="Output directory to result files",
        default='./eval_data/eval_out')
    arg_parser.add_argument(
        "--tracker_fol", help="Directory to tracker input file",
        default='./eval_data/trackers')
    arg_parser.add_argument(
        "--seq_ini", help="Path to ini file",
        default='./eval_data/seqinfo.ini')
    arg_parser.add_argument(
        "--seqmap_file", help="Path to gt file name seq.",
        default='./eval_data/gt/seqmaps/MOT17-train.txt')
    return arg_parser.parse_args()


if __name__ == '__main__':
    freeze_support()
    args_eval = create_eval_arg_parser()
    # Command line interface:
    default_eval_config = trackeval.Evaluator.get_default_eval_config()
    default_eval_config['DISPLAY_LESS_PROGRESS'] = False
    default_dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config(args_eval.gt_fol,
                                                                                             args_eval.tracker_fol,
                                                                                             args_eval.out_fol,
                                                                                             args_eval.seq_ini,
                                                                                             args_eval.seqmap_file)
    default_metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity']}


    config = {**default_eval_config, **default_dataset_config, **default_metrics_config}  # Merge default configs

    eval_config = {k: v for k, v in config.items() if k in default_eval_config.keys()}
    dataset_config = {k: v for k, v in config.items() if k in default_dataset_config.keys()}
    metrics_config = {k: v for k, v in config.items() if k in default_metrics_config.keys()}

    # Run code
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(args_eval.gt_fol, args_eval.tracker_fol, args_eval.out_fol,
                                                         args_eval.seq_ini, args_eval.seqmap_file,
                                                         dataset_config)]
    # (args_eval.gt_fol,args_eval.tracker_fol,args_eval.out_fol,args_eval.seq_ini,args_eval.seqmap_file,dataset_config)]
    metrics_list = []
    for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity,
                   trackeval.metrics.VACE]:
        if metric.get_name() in metrics_config['METRICS']:
            metrics_list.append(metric())
    if len(metrics_list) == 0:
        raise Exception('No metrics selected for evaluation')
    evaluator.evaluate(dataset_list, metrics_list)


"""parser = argparse.ArgumentParser()
    for setting in config.keys():  # merged settings
        if type(config[setting]) == list or type(config[setting]) == type(None):
            parser.add_argument("--" + setting, nargs='+')
        else:
            parser.add_argument("--" + setting)


args = parser.parse_args().__dict__
    for setting in args.keys():
        if args[setting] is not None:
            if type(config[setting]) == type(True):
                if args[setting] == 'True':
                    x = True
                elif args[setting] == 'False':
                    x = False
                else:
                    raise Exception('Command line parameter ' + setting + 'must be True or False')
            elif type(config[setting]) == type(1):
                x = int(args[setting])
            elif type(args[setting]) == type(None):
                x = None
            elif setting == 'SEQ_INFO':
                x = dict(zip(args[setting], [None] * len(args[setting])))
            else:
                x = args[setting]
            config[setting] = x"""