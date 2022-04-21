import cv2
import time
import os
import tensorflow as tf
import numpy as np
import os.path as osp
import pandas as pd
import argparse



# CUDA_VISIBLE_DEVICES = 0
# tf.config.experimental.set_memory_growth = True
tf.compat.v1.disable_v2_behavior()


def mkdirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)


# Preprocessing input data(bounding box labels)
class CSVReader(object):

    def __init__(self, video_path_in, video_path_out, csv_path, track_res_path):
        self._video_path_in = video_path_in
        self._video_path_out = video_path_out
        self._track_res_path = track_res_path
        self._columns = ["track_id","frame", "xmin", "ymin","xmax","ymax","class_id"]
        self._csv_path = csv_path
        self.set_vid_frame_info()
        self.sort_gtbox()

    # prepare video output write format
    def set_vid_frame_info(self):
        try:
            cap = cv2.VideoCapture(int(self._video_path_in))
        except:
            cap = cv2.VideoCapture(self._video_path_in)
        self._fps = cap.get(cv2.CAP_PROP_FPS)
        self._video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._video_frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self._fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    def vid_writer(self):
        dir_out = str(self._video_path_out)
        mkdirs(dir_out)
        file_out_name = osp.join(dir_out + "/" + "track_out_dist08_0780id18156-chi950515.mp4")
        out = cv2.VideoWriter(file_out_name, self._fourcc, self._fps, (self._video_width, self._video_height))
        return out

    def write_track_result(self, results):

        f = open(self._track_res_path, 'w')
        for row in results:
            print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
                row[0], row[1], row[2], row[3], row[4], row[5]), file=f)  # changed sequence fid,tid,bbox,..

    # sort bounding boxes labels from original yolo detection file according to frame index, save as npy file
    def sort_gtbox(self):
        csv = pd.read_csv(self._csv_path,
                          usecols=['ID', 'Time', 'screen_bbox_x1', 'screen_bbox_y1', 'screen_bbox_x2', 'screen_bbox_y2',
                                   'Class'], engine='python')
        csv.columns = self._columns  # [1796386 rows x 7 columns] frames 29367
        gt = csv.to_numpy()
        self.gt_sorted = np.asarray(gt)

    # create track id index array list to locate all track id in each frame
    def get_bbox(self, frame_num):
        total = self.gt_sorted[self.gt_sorted[:, 1] == frame_num]
        bbox = total[:, 2:6]
        c = total[:, 6]
        #  tid = total[:, 0]
        return bbox, c, total

    # convert bbox format from (xmin,ymin,xmax,ymax) to (xmin,ymin,width,height)
    def bbox_tlwh(self, bboxes):
        for box in bboxes:
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])
            width = x2 - x1
            height = y2 - y1
            box[0], box[1], box[2], box[3] = x1, y1, width, height
        return bboxes

    def loading_percent(self, frame_num):
        percent = float(100*frame_num/int(self._video_frame_count))
        return percent


# get csv file paths, names; get video paths, names for Crop_bbox_for_train.py
def read_train_data_split(is_vid, in_directory):
    mkdirs(in_directory)
    in_filenames, in_file_paths = [], []
    for (dirpath, dirnames, filenames) in os.walk(in_directory):
        for filename in sorted(filenames):
            path = osp.join(dirpath + "/", filename)
            filename_base, ext = os.path.splitext(filename)
            #filename_base = filename_base.split('_')[0]
            if is_vid and ext != ".mp4":
                print("is not a video!")
                continue
            if is_vid == False and ext != ".csv":
                print("is not a csv file!")
            in_filenames.append(filename_base)
            in_file_paths.append(path)
    return in_file_paths, in_filenames


# get video, csv file path for tracker_app.py
def read_single_vid_csv(is_vid, in_directory):
    mkdirs(in_directory)
    for (dirpath, dirnames, filenames) in os.walk(in_directory):
        in_path = osp.join(dirpath + "/", os.listdir(in_directory)[0])
        filename_base, ext = os.path.splitext(os.listdir(in_directory)[0])
        if is_vid and ext != ".mp4":
            print("is not a video!")
            continue
        if is_vid == False and ext != ".csv":
            print("is not a csv file!")
    return in_path, filename_base


def str_arr_to_int(class_str):
    np.place(class_str, class_str=='Car', '1')
    np.place(class_str, class_str=='Truck', '2')
    np.place(class_str, class_str== 'Van', '3')
    np.place(class_str, class_str == 'Bus', '4')


def class_to_int(c_str):
    if c_str == 'Car':
        c = 1
    if c_str == 'Truck':
        c = 2
    if c_str == 'Van':
        c = 3
    else:   # Bus
        c = 4
    return c


# csv has 0-29367 frames.  1629491 items
# define data path and model path for single video
def create_default_arg_parser():
    arg_parser = argparse.ArgumentParser(
        description="csv_reader check")
    arg_parser.add_argument(
        "--video_path_in", help="Path to one DJI input video file.",
        default='./tracker_input')
    arg_parser.add_argument(
        "--video_path_out", help="Path to tracked video file.",
        default='./output/DJI_out.mp4')
    arg_parser.add_argument(
        "--csv_path", help="Path to one csv file.",
        default='./dataset/tracking_real_new.csv')
    return arg_parser.parse_args()


if __name__ == '__main__':
    import tracker_app
    args_tracker = tracker_app.create_tracker_arg_parser()
    ds = CSVReader(args_tracker.video_path_in, args_tracker.video_dir_out, args_tracker.csv_path_in,
                   track_res_path=args_tracker.track_res_path)
