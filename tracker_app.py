import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import csv_reader as reader

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
        default='./track_res_iou0780_dis08-chi95_reid18156.csv')
    arg_parser.add_argument(
        "--csv_path_in", help="Path to csv file.",
        default='./dataset/tracker_input/csv/01_TrackTrain.csv')
    arg_parser.add_argument(
        "--video_dir_out", help="Path to tracking video directory.",
        default='./tracker_output')
    arg_parser.add_argument(
        "--model_path", help="Path to reid model weights file.",
        default='./model_data/freeze_dji-18156.pb')
    arg_parser.add_argument(
        "--show_video", help="True for show tracking result in every frame, false for not show",
        default=False)
    arg_parser.add_argument(
        "--max_age", help="Define max age with integer number",
        default=80)
    arg_parser.add_argument(
        "--max_iou", help="Define max_iou value from 0 to 1",
        default=0.7)
    arg_parser.add_argument(
        "--max_cosine_distance", help="Define max_cosine_distance from 0 to 1",
        default=0.8)

    return arg_parser.parse_args()


def main():
    # definition of the parameters
    # get video paths, names
    nn_budget = None
    args_tracker = create_tracker_arg_parser()
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", args_tracker.max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric=metric, max_iou_distance=args_tracker.max_iou, max_age=int(args_tracker.max_age))
    dataset = reader.CSVReader(args_tracker.video_path_in, args_tracker.video_dir_out, args_tracker.csv_path_in,
                               args_tracker.track_res_path)
    
    try:
        vid = cv2.VideoCapture(int(args_tracker.video_path_in))
    except:
        vid = cv2.VideoCapture(args_tracker.video_path_in)

    out = dataset.vid_writer()

    track_result = []
    # prepare image encoder for reid model
    encoder = gdet.create_box_encoder(model_filename=args_tracker.model_path, batch_size=1)

    # load label,bounding boxes files as numpy array
    # for loop each video frame

    for frame_num in range(-1, int(dataset._video_frame_count)):
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
        boxes, class_names, totals = dataset.get_bbox(frame_num)
        original_h, original_w, _ = frame.shape
        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        bboxes = dataset.bbox_tlwh(boxes)
        # extract reid features from cosine_metric_learning model, batch_size = 1 for one frame each time
        features = gdet.generate_detections(encoder, bboxes, frame)
        detections_out = [Detection(bbox, class_name, feature) for bbox, class_name, feature in
                          zip(bboxes, class_names, features)]
        # initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # Call the tracker
        tracker.predict()
        tracker.update(detections_out)
        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            tracker_class_name = track.get_class()
            # draw bbox on screen
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            # draw bounding boxes in each frame with line width 2
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1] - 30)),
                          (int(bbox[0]) + (len('class') + len(str(track.track_id))) * 17, int(bbox[1])), color, -1)
            cv2.putText(frame, tracker_class_name + "-" + str(track.track_id), (int(bbox[0]), int(bbox[1] - 10)), 0, 0.75,
                        (255, 255, 255), 2)
            class_int = reader.class_to_int(tracker_class_name)
            track_result.append([int(frame_num), int(track.track_id), bbox[0], bbox[1], bbox[2], bbox[3], class_int])

            # if frame read successfully print details about each track
            if return_value:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(
                    str(track.track_id),
                    tracker_class_name, (
                        int(bbox[0]),
                        int(bbox[1]),
                        int(bbox[2]),
                        int(bbox[3]))))

        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps, flush=True)
        end = time.time()
        print("Dauer:")
        print(end - start_time)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if args_tracker.show_video:
            cv2.imshow("Output Video", result)
        out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

    dataset.write_track_result(track_result)


if __name__ == '__main__':
    main()

