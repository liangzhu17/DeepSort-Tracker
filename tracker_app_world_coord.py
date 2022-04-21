import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
import os.path as osp
import pandas as pd
from scipy.optimize import fsolve
from decouple import config
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
# deep sort_wdet imports
from deep_sort_world_coord import nn_matching
from deep_sort_world_coord.detection import Detection
from deep_sort_world_coord.tracker import Tracker
from tools import generate_detections as gdet


# CUDA_VISIBLE_DEVICES = 0
# tf.config.experimental.set_memory_growth = True
tf.compat.v1.disable_v2_behavior()

s_fid = config('s_fid2')
e_fid = config('e_fid2')


def mkdirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)


# Preprocessing input data(bounding box labels)
class CSVReader(object):

    def __init__(self, video_path_in, video_path_out, csv_in, track_res_path):
        self._video_path_in = video_path_in
        self._video_path_out = video_path_out
        self._track_res_path = track_res_path
        self._columns = ["frame", "cam_x1", "cam_y1", "cam_x2", "cam_y2","w_x1", "w_y1", "w_x2", "w_y2", "w_rot", "x_c",
                         "y_c", "class_id"]
        self._csv_in = csv_in
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
        base_out = str(self._video_path_out)
        dir_out = osp.join(base_out + "/" + s_fid + "_" + e_fid)
        mkdirs(dir_out)
        p_out = osp.join(dir_out, "track_out_079008-0614.mp4")
        out = cv2.VideoWriter(p_out, self._fourcc, self._fps, (self._video_width, self._video_height))
        return out

    def write_track_res_tlbr(self, results):
        dir_base = str(self._track_res_path)
        dir_out = osp.join(dir_base + "/" + s_fid + "_" + e_fid)
        mkdirs(dir_out)
        p_out = osp.join(dir_out, "079010_std_adap_poliou_track_res_w_tlbr_0616.csv")
        f = open(p_out, 'w')
        for row in results:
            print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
                row[0], row[1], row[2], row[3], row[4], row[5]), file=f)  # changed sequence fid,tid,bbox,..

    def write_track_res_polygon(self, results):
        dir_base = str(self._track_res_path)
        dir_out = osp.join(dir_base + "/" + s_fid + "_" + e_fid)
        mkdirs(dir_out)
        p_out = osp.join(dir_out, "track_res_w_poliou_tlwh079010poly_0616.csv")
        f = open(p_out, 'w')
        for row in results:
            print('%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
                row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9]), file=f)  # changed sequence fid,tid,bbox,..

    # sort bounding boxes labels from original yolo detection file according to frame index, save as npy file
    def sort_gtbox(self):
        csv = pd.read_csv(self._csv_in,
                          usecols=['Time', 'screen_bbox_x1', 'screen_bbox_y1', 'screen_bbox_x2', 'screen_bbox_y2',
                                   'world_bbox_x1', 'world_bbox_y1', 'world_bbox_x2', 'world_bbox_y2',
                                   'world_bbox_length', 'world_bbox_width', 'world_bbox_rotation',
                                   'world_center_x', 'world_center_y', 'Class'], engine='python')
        #csv.columns = self._columns  # [1796386 rows x 13 columns] frames 29367 # [:, 0:12]
        gt = csv.to_numpy()
        self.gt = np.asarray(gt)

    # create track id index array list to locate all track id in each frame
    def get_bbox(self, frame_num):
        total = self.gt[self.gt[:, 0] == frame_num] # 0 ID
        bbox_cam = total[:, 1:5]
        bbox_w_xy3xy1 = total[:, [7, 8, 5, 6]]  # first xmin xy3, then xmax xy1
        bbox_w_xy3lh = total[:, 7:11]    #
        rot_alpha = total[:, 11]
        bbox_w_xyclh = total[:, [12, 13, 9, 10]]
        #h = total[:, 9]
        #w = total[:, 10]
        c = total[:, 14]

        return bbox_cam, bbox_w_xy3xy1, bbox_w_xy3lh, bbox_w_xyclh, c, rot_alpha

    def alpha_convert(self, alpha_arr): # change alpha to 0 - pi/2
        ret = alpha_arr.copy()
        for i in range(len(ret)):
            ret[i] = 180.0 - (ret[i] / np.pi * 180)
        return ret

    # convert bbox format from (xmin,ymin,xmax,ymax) ----> (x3, y3, w, h) ---> (xmin,ymin,width,height)
    def bbox_tlwh(self, bboxes):

        for box in bboxes:   # (xmin,ymin,xmax,ymax)
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])
            width = x2 - x1
            height = y2 - y1
            box[0], box[1], box[2], box[3] = x1, y1, width, height
        return bboxes

    def loading_percent(self, frame_num):
        percent = float(100 * frame_num / int(self._video_frame_count))
        return percent


def class_to_int(c_str):
    if c_str == 'Car':
        c = 1
    elif c_str == 'Truck':
        c = 2
    elif c_str == 'Van':
        c = 3
    else:  # Bus
        c = 4
    return c

"""Test gap frames cuts"""

# define data path and reid model path
def create_tracker_arg_parser():
    """Parse command line arguments.
        """
    arg_parser = argparse.ArgumentParser(
        description="tracker prepare")
    arg_parser.add_argument(
        "--video_path_in", help="Path to DJI input video file with .MOV or.mp4.",
        default='./dataset/tracker_input/vid/DJI_in.mp4')
    arg_parser.add_argument(
        "--track_res_path", help="Path to track results csv File",
        default='./dataset/track_res_out/world_0617_test')  # no budget
    arg_parser.add_argument(
        "--csv_in", help="Path to csv file.",
        default='./dataset/tracker_input/csv/01_TrackTrain_fps50.csv')
    arg_parser.add_argument(
        "--video_dir_out", help="Path to tracking video directory.",
        default='./tracker_output/w_det_polyxy4_1.0890_0617')
    arg_parser.add_argument(
        "--model_path", help="Path to reid model weights file.",
        default='./model_data/freeze_dji-18156.pb')
    arg_parser.add_argument(
        "--show_video", help="True for show tracking result in every frame, false for not show",
        default=False)
    arg_parser.add_argument(
        "--max_cos_dis", help="Max cosine distance between features, range from 0 to 2?",
        default=0.8)
    return arg_parser.parse_args()


def main():
    # definition of the parameters
    # get video paths, names
    nn_budget = None

    max_iou = float(config('max_iou2'))
    max_dist = float(config('max_cos_dist2'))
    max_age = int(config('max_age2'))
    s_fid = int(config('s_fid2'))
    e_fid = int(config('e_fid2'))
    n_init = int(config('n_init2'))
    args_tracker = create_tracker_arg_parser()
    # calculate cosine distance metric
    # can also be euclidean "euclidean"
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_dist, nn_budget)

    # initialize tracker
    tracker = Tracker(metric=metric, max_iou_distance=max_iou, max_age=max_age, n_init=n_init)
    dataset = CSVReader(args_tracker.video_path_in, args_tracker.video_dir_out, args_tracker.csv_in,
                               args_tracker.track_res_path)
    try:
        vid = cv2.VideoCapture(int(args_tracker.video_path_in))
    except:
        vid = cv2.VideoCapture(args_tracker.video_path_in)

    track_result = []
    track_res_pol = []
    # prepare image encoder for reid model
    encoder = gdet.create_box_encoder(model_filename=args_tracker.model_path, batch_size=1)

    # load label,bounding boxes files as numpy array
    # for loop each video frame

    for frame_num in range(s_fid-1, e_fid):
        frame_num += 1
        vid.set(cv2.CAP_PROP_POS_FRAMES, int(frame_num))
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            print('Video has ended or failed, try a different video format!')
            break

        print('Frame #: ', frame_num, flush=True)
        percent = dataset.loading_percent(frame_num)
        print("Tracking percent: ", percent, "%", flush=True)
        # Time for FPS
        start_time = time.time()

        cam_bboxes, world_bboxes_tlbr, world_bboxes_tlwh, world_bboxes_xywh, class_names, rot_alpha = dataset.get_bbox(frame_num)
        original_h, original_w, _ = frame.shape
        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        # measurements and wh_compare
        cam_tlwh_boxes = dataset.bbox_tlwh(cam_bboxes)
        alphas_converted = dataset.alpha_convert(rot_alpha)

        # extract reid features from cosine_metric_learning model, batch_size = 1 for one frame each time
        features = gdet.generate_detections(encoder, cam_tlwh_boxes, frame)
        detections_out = [Detection(cam_tlwh_box, world_tlwh_box, world_tlbr_box, world_xywh_box, alpha_converted, class_name, feature)
                          for cam_tlwh_box, world_tlwh_box, world_tlbr_box, world_xywh_box, alpha_converted, class_name, feature in
                          zip(cam_tlwh_boxes, world_bboxes_tlwh, world_bboxes_tlbr, world_bboxes_xywh,
                              alphas_converted, class_names, features)]

        # Call the tracker
        tracker.predict()
        tracker.update(detections_out)
        # update tracks
        for track in tracker.tracks:
            if track.time_since_update > 1:
                continue

            bbox_tlbr = track.to_w_tlbr()
            # output the 4 polygon track res xy3214 tl,....br,....
            bbox_pol = track.to_w_polygon()
            tracker_class_name = track.get_class()
            class_int = class_to_int(tracker_class_name)
            track_result.append([int(frame_num)+1, int(track.track_id),
                                 bbox_tlbr[0], bbox_tlbr[1], bbox_tlbr[2], bbox_tlbr[3], class_int])

            track_res_pol.append([int(frame_num) + 1, int(track.track_id),
                                 bbox_pol[0][0], bbox_pol[0][1], bbox_pol[1][0],bbox_pol[1][1],bbox_pol[2][0], bbox_pol[2][1], bbox_pol[3][0],bbox_pol[3][1]])

            # first world x3y3, then world x1y1

        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps, flush=True)
        end = time.time()
        print("Dauer:")
        print(end - start_time)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

    dataset.write_track_res_tlbr(track_result)
    dataset.write_track_res_polygon(track_res_pol)

if __name__ == '__main__':
    main()
