import os
import os.path as osp
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import numpy as np
import glob
import argparse
import re
import csv_reader as reader


def mkdirs(d):
    if not os.path.isdir(d):
        os.makedirs(d)


# define data path and model path
def create_crop_arg_parser():
    arg_parser = argparse.ArgumentParser(description="crop_preprocessing")
    arg_parser.add_argument("--video_dir_in", help="Path to DJI input video directory.",default='./dataset/vid_input')
    arg_parser.add_argument("--crop_dir_out", help="Path to output directory for cropped images.",
                            default='./dataset/data_crop_prep_try/train')
    arg_parser.add_argument("--csv_dir_in", help="Path to DJI dataset directory.",
                            default='./dataset/csv_input')
    return arg_parser.parse_args()


# crop bounding boxes from input video frames each 10 frames, for cosine_metric_learning, images saved as jpeg
def crop_bbox(image, frame_num, tot_csv_labels, output_dir, filename_base):
    if frame_num % 10 == 0:  # each 10 frames, crop bounding boxes
        for total in tot_csv_labels:
            tid = total[0]
            fid = total[1]
            x1 = int(total[2])
            y1 = int(total[3])
            x2 = int(total[4])
            y2 = int(total[5])
            c = total[6]
            crop = image[y1:y2, x1:x2, :]
            # fid // 4500 for maximal 32000 items in one linux folder
            # set camera index, default one camera c01
            dir = osp.join(output_dir + "/" + str(filename_base) + "/" + str(fid // 4600))
            mkdirs(dir)
            cv2.imwrite(dir + "/" + str(tid).split(".")[0] + "_c01_" + c + "_" + str(fid).split(".")[0] + ".jpg", crop)


def main():

    args_crop = create_crop_arg_parser()
    dataset = reader.CSV_reader(args_crop.video_path_in, args_crop.video_path_out, args_crop.csv_path)
    vid_file_paths, vid_filenames = reader.read_train_data_split(is_vid=True, in_directory=args_crop.video_dir_in)
    print(vid_file_paths)
    csv_file_paths, csv_filenames = reader.read_train_data_split(False, args_crop.csv_dir_in)
    print("csv_file_paths:", csv_file_paths)

    for i in range(0, len(vid_file_paths)):
        if len(vid_file_paths) != len(csv_file_paths):
            print("Numbers of Input videos and csv files don't match!")
            break
        vid_path_in = vid_file_paths[i]
        csv_path_in = csv_file_paths[i]
        dataset = reader.CSV_reader(vid_path_in, args_crop.crop_dir_out, csv_path_in)
        # begin video capture
        try:
            vid = cv2.VideoCapture(int(vid_path_in))
        except:
            vid = cv2.VideoCapture(vid_path_in)

        frame_num = -1
        # load label,bounding boxes files as numpy array
        # while video is running
        while True:
            return_value, frame = vid.read()
            if return_value:
                image = np.asarray(frame)
            else:
                print('Video has ended or failed, try a different video format!')
                break
            frame_num += 1
            print('Frame #: ', frame_num)
            _, _, totals = dataset.get_bbox(frame_num)
            crop_bbox(image, frame_num, totals, args_crop.crop_dir_out, vid_filenames[i])
            percent = dataset.loading_percent(frame_num)
            print("Process video: ", vid_filenames[i], ", frame percent:", percent, "%")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()