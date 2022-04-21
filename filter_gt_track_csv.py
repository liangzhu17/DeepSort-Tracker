import time
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import argparse
from scipy.optimize import linear_sum_assignment

tf.compat.v1.disable_v2_behavior()

def mkdirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)

# convert bbox format from (xmin,ymin,xmax,ymax) to (xmin,ymin,width,height)
def bbox_xywh(bboxes):
    for box in bboxes:
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])
        width = x2 - x1
        height = y2 - y1
        box[0], box[1], box[2], box[3] = x1, y1, width, height
    return bboxes


def iou(bbox, candidates):
    """Computer intersection over union.

    Parameters
    ----------
    bbox : ndarray
        A bounding box in format `(top left x, top left y, width, height)`.
    candidates : ndarray
        A matrix of candidate bounding boxes (one per row) in the same format
        as `bbox`.

    Returns
    -------
    ndarray
        The intersection over union in [0, 1] between the `bbox` and each
        candidate. A higher score means a larger fraction of the `bbox` is
        occluded by the candidate.

    """
    bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, :2] + candidates[:, 2:]

    tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
               np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
    br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
               np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
    wh = np.maximum(0., br - tl)

    area_intersection = wh.prod(axis=1)  # return product of valued on axis 1
    area_bbox = bbox[2:].prod()
    area_candidates = candidates[:, 2:].prod(axis=1)
    return area_intersection / (area_bbox + area_candidates - area_intersection)


# Preprocessing input data(bounding box labels)
class CSVReader(object):

    def __init__(self, gt_path_in, track_res_in, gt_filtered_out, track_filtered_out):

        self._track_res_in = track_res_in
        self._columns = ["frame", "track_id", "xmin", "ymin","xmax","ymax","class_id"]
        self._gt_filtered_out = gt_filtered_out
        self._track_filtered_out = track_filtered_out
        self._gt_path_in = gt_path_in  # original csv file
        self.gtbox()
        self.get_global_match_list()
        self.get_gap_list()
        self.get_gt_rm_idx()
        #self.get_track_rm_idx()

    def gtbox(self):
        csv = pd.read_csv(self._gt_path_in,
                          usecols=['Time', 'ID', 'screen_bbox_x1', 'screen_bbox_y1', 'screen_bbox_x2', 'screen_bbox_y2',
                                   'Class'], engine='python')
        csv.columns = self._columns  # [1796386 rows x 7 columns] frames 29367
        gt = csv.to_numpy()
        self.gt_s = np.asarray(gt)
        self.gt_s = self.gt_s[:, [1, 0, 2, 3, 4, 5, 6]]
        self.frame_count = np.amax(self.gt_s[:, 0], axis=0)

    # np.set_printoptions(edgeitems=150)
    # print("self.gt_s:", self.gt_s)
    def get_global_match_list(self):
        # idx = np.lexsort(self.gt_s.T[1:2, :])
        # self.gt_s = self.gt_s[idx, :]
        #self.gt_s = pd.read_csv(self._gt_path_in, header=None, engine='python')
        #self.gt_s = np.asarray(self.gt_s)  # not sorted
        self.frame_count = np.amax(self.gt_s[:, 0], axis=0)
        self.track_csv = pd.read_csv(self._track_res_in, header=None, engine='python')
        self.track_res = self.track_csv.to_numpy()
        self.track_res = np.asarray(self.track_res)
        match_gt_global_idx = []
        match_tracker_global_idx = []
        gt_fid_match_list = []
        gt_tid_match_list = []
        #track_fid_match_list = []
        track_tid_match_list = []

        for fid in range(int(self.frame_count)+1):  # changed frame index, start from fid=1
            print("iou calculation process: ", fid/(int(self.frame_count)+1)*100, "%", flush=True)
            gt_totals = self.gt_s[self.gt_s[:, 0] == fid]
            gt_bboxes = bbox_xywh(gt_totals[:, 2:6])
            tracker_totals = self.track_res[self.track_res[:, 0] == fid+1]
            tracker_bboxes = bbox_xywh(tracker_totals[:, 2:6])
            iou_matrix = []
            for gt_box in gt_bboxes:
                iou_single = iou(gt_box, tracker_bboxes)
                if len(iou_single)==0:
                    break
                else:
                    if len(iou_matrix) == 0:
                        iou_matrix = iou_single
                    else:
                        iou_matrix = np.vstack((iou_matrix, iou_single))

            if np.shape(iou_matrix)[0] !=0:
                match_rows, match_cols = linear_sum_assignment(iou_matrix)  # same frame id, all bounding boxes (local index)
                # filter match_rows
                for i in range(len(match_rows)):
                    r = match_rows[i]
                    c = match_cols[i]
                    tid_gt = int(gt_totals[r, 1])
                    fid_r = int(gt_totals[r, 0])

                    tid_t = int(tracker_totals[c, 1])
                    fid_c = int(tracker_totals[c, 0])

                    if fid_r != fid:
                        print("fid_x=", fid_r, "!= fid:", fid, "input data errors in frame id")
                        break
                    if fid_c != fid+1:    # index difference is 1, gt and tracker
                        print("fid in tracker=", fid_c, "!= fid in gt:", fid+1)
                        break

                    gt_fid_match_list.append(fid)
                    gt_tid_match_list.append(tid_gt)
                    match_gt_global_idx.append(
                        np.where(np.logical_and(self.gt_s[:, 0] == fid, self.gt_s[:, 1] == tid_gt)))  # global row num

                    track_tid_match_list.append(tid_t)
                    match_tracker_global_idx.append(
                        np.where(np.logical_and(self.track_res[:, 0] == fid+1,
                                                self.track_res[:, 1] == tid_t)))  # global tracker num

        match_gt_global_idx = np.squeeze(match_gt_global_idx)
        match_tracker_global_idx = np.squeeze(match_tracker_global_idx)

        np.set_printoptions(edgeitems=50, linewidth=50)
        f1 = open('./gt_tid_all_matchlist.csv', 'w')
        for row in gt_tid_match_list:
            print('%d' % (row), file=f1)
        f2 = open('./gt_fid_all_matchlist.csv', 'w')
        for row in gt_fid_match_list:
            print('%d' % (row), file=f2)
        f3 = open('./match_gt_global_idx_all.csv', 'w')
        for row in match_gt_global_idx:
            print('%d' % (row), file=f3)

        self.gt_global_match_idx = match_gt_global_idx
        self.tracker_global_match_idx = match_tracker_global_idx
        self.gt_fid_match_list = gt_fid_match_list
        self.gt_tid_match_list = gt_tid_match_list

        self.track_tid_match_list = track_tid_match_list

    def get_gap_list(self):        # 87s
        self.rm_gap_tid_dict = {}  # maps gap tid to global row numbers
        self.tid_gap_idx = []
        tid_unique = np.unique(self.gt_s[:, 1])
        tid_unique = tid_unique.astype(np.int)
        for tid in tid_unique:
            gt_same_tid = self.gt_s[self.gt_s[:, 1] == tid]
            fid_list = gt_same_tid[:, 0]  # all fid for one tid

            for i in range(len(fid_list) - 1):  # each pair (fid, tid), loop all fids for one tid
                if fid_list[i + 1] != fid_list[i] + 1:
                    if not (tid in self.rm_gap_tid_dict.keys()):
                        self.tid_gap_idx.append(tid)  # store all gap track ids(occlusion)
                        gap_tid_global_row = np.squeeze(np.where(self.gt_s[:, 1] == tid))
                        self.rm_gap_tid_dict[tid] = gap_tid_global_row

        np.set_printoptions(edgeitems=50, linewidth=50)
        #print("self.tid_gap_idx: ", self.tid_gap_idx)
        #print("dict_", self.rm_gap_tid_dict)

    def get_gt_rm_idx(self):
        self.rm_gt_idx = []
        tid_unique = np.unique(self.gt_s[:, 1])
        tid_unique = tid_unique.astype(np.int)
        counter = 0
        temporal_tid = []
        tid_rm_possible = []
        t = []
        for tid in tid_unique:
            counter = counter + 1
            print("get gt rm index process: ", 100 * (counter / int(np.shape(tid_unique)[0])), "%", flush=True)
            gt_same_tid = self.gt_s[self.gt_s[:, 1] == tid]
            fid_list = gt_same_tid[:, 0]  # all fid for one tid

            if not (tid in self.rm_gap_tid_dict.keys()):
                gt_suc_row = np.squeeze(np.where(self.gt_s[:, 1] == tid))
                if len(self.rm_gt_idx)==0:
                    self.rm_gt_idx = gt_suc_row
                self.rm_gt_idx = np.hstack((self.rm_gt_idx, gt_suc_row))

            else:  # successive

                for i in range(len(fid_list) - 1):  # each pair (fid, tid), loop all fids for one tid
                    # in all global gt row index for (fid, tid) and (fid_next, tid)
                    if tid == 4 and i == 0:
                        st = time.time()
                        t.append(st)
                    if tid == 8 and i == 0:
                        et = time.time()
                        t.append(et)
                        print("tid4 f0 to f last, time is", t[1] - t[0])

                    if fid_list[i + 1] == np.amax(fid_list, axis=0):  # tid in last fid, endet
                        tid_rm_possible = np.unique(tid_rm_possible)
                        temporal_tid = np.unique(temporal_tid)
                        if tid in self.rm_gap_tid_dict.keys():  # gap tids
                            gt_global_match_next_idx = np.where(
                                np.logical_and(self.gt_fid_match_list == fid_list[i + 1],
                                               self.gt_tid_match_list == tid))
                            gt_s_match_next_idx = self.gt_global_match_idx[gt_global_match_next_idx]
                            if np.any(temporal_tid == tid) != 0 and len(
                                    gt_s_match_next_idx) != 0:  # matched in last fid
                                tid_rm_possible = np.hstack((tid_rm_possible, tid))

                    else:  # not last fid, where tid exists

                        #  all gap cases
                        if tid in self.rm_gap_tid_dict.keys():
                            gt_global_match_next_idx = np.where(
                                np.logical_and(self.gt_fid_match_list == fid_list[i + 1],
                                               self.gt_tid_match_list == tid))
                            #gt_global_match_idx = np.where(
                            #    np.logical_and(self.gt_fid_match_list == fid_list[i], self.gt_tid_match_list == tid))
                            #gt_s_match_idx = self.gt_global_match_idx[gt_global_match_idx]
                            gt_s_match_next_idx = self.gt_global_match_idx[gt_global_match_next_idx]

                            if len(gt_s_match_next_idx) != 0:  # found in match_list
                                # print("gt_global_match_idx", gt_global_match_idx)
                                """if not np.all(self.gt_s[gt_s_match_idx] == self.gt_s[gt_to_rm]):
                                    print("Errors of index: gt_global_match_idx using match_idx_global maps to",
                                          self.gt_s[gt_s_match_idx])
                                    print(" are different from gt_to_rm", self.gt_s[gt_to_rm])
                                    print("fid:", fid_list[i], "tid", tid)"""
                                # else: gap found, occlusion case, but tid matched in fid and fid+n dont remove

                                tid_rm_possible = np.hstack(
                                    (tid_rm_possible, tid))  # matched again (or always matched) should not evaluate
                                if np.any(temporal_tid == tid):
                                    temporal_tid = np.delete(temporal_tid, np.where(temporal_tid == tid))
                                break

                            elif len(gt_s_match_next_idx) == 0:  # fid matched, fid+1, tid not matched anymore, FN
                                if not (np.any(tid_rm_possible == tid)):
                                    temporal_tid = np.hstack((temporal_tid, tid))


        self.rm_gt_idx = np.unique(self.rm_gt_idx)

        # print("__gt_rm_row after unique of begin", self.rm_gt_idx, np.shape(self.rm_gt_idx))  (1430,)
        tid_rm_possible = np.unique(tid_rm_possible)
        print("tid_rm_possibel:", tid_rm_possible)

        for pos_tid in tid_rm_possible:
            gt_s_row = np.squeeze(np.where(self.gt_s[:, 1] == pos_tid))
            self.rm_gt_idx = np.hstack((self.rm_gt_idx, gt_s_row))
            #print("self gt  rm list_ ", self.rm_gt_idx, np.shape(self.rm_gt_idx))
            #print("__gt_s row num", gt_s_row, np.shape(gt_s_row))

    def get_track_rm_idx(self):
        self.remove_track_idx = []

        counter = 0
        for rm_idx in self.rm_gt_idx:
            counter+=1
            print("track to be removed index processing: ", 100 * (counter / len(self.rm_gt_idx)), "%",flush=True)
            gt_s_row = self.gt_s[rm_idx]  # ---> match_row, match,col
            if np.any(self.gt_fid_match_list == gt_s_row[0]) and np.any(self.gt_tid_match_list==gt_s_row[1]):
                global_match_idx = np.where(np.logical_and(self.gt_fid_match_list == gt_s_row[0], self.gt_tid_match_list == gt_s_row[1]))
                global_match_idx = np.squeeze(global_match_idx)
                if len(np.shape(global_match_idx))==1:
                    print("after squeeze", global_match_idx, np.shape(global_match_idx))
                else:
                    self.tracker_global_match_idx[global_match_idx] = np.squeeze(
                        self.tracker_global_match_idx[global_match_idx])  # get row num in track_res
                    self.remove_track_idx.append(self.tracker_global_match_idx[global_match_idx])

    def write_filter_gt_result(self):
        self.rm_gt_idx = np.unique(self.rm_gt_idx)
        self.filtered_gt = np.delete(self.gt_s, self.rm_gt_idx, axis=0)

        f = open(self._gt_filtered_out, 'w')
        for row in self.filtered_gt:
            print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,1,1' % (
                row[0]+1, row[1], row[2], row[3], row[4], row[5]), file=f)  # second is tid

    def write_filter_track_result(self):
        filtered_track_res = np.delete(self.track_res, self.remove_track_idx, axis=0)

        f = open(self._track_filtered_out, 'w')
        for row in filtered_track_res:
            print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
                row[0], row[1], row[2], row[3], row[4], row[5]), file=f)


    def get_max_min_label(self):
        l_unique = np.unique(self.gt_s[:, 0])
        l = l_unique.astype(np.int)
        max_label = np.amax(l, axis=0)
        min_label = np.amin(l, axis=0)
        return max_label, min_label


def create_filter_arg_parser():
    """Parse command line arguments."""
    arg_parser = argparse.ArgumentParser(
        description="tracker prepare")

    arg_parser.add_argument(
        "--gt_path_in", help="Path to csv file.",
        default='./dataset/01_TrackTrain.csv')
    arg_parser.add_argument(
        "--track_res_in", help="Path to track results csv File",
        default='./dataset/track_res29368_new.csv')
    arg_parser.add_argument(
        "--gt_filtered_out", help="Path to gt filtered csv File",
        default='./dataset/gt_filtered_all.csv')
    arg_parser.add_argument(
        "--track_filtered_out", help="Path to track filtered results csv File",
        default='./dataset/track_res_filtered_all.csv')
    arg_parser.add_argument(
        "--gt_eval", help="Output directory to ground truth file.",
        default='./eval_data')
    arg_parser.add_argument(
        "--tracker_eval", help="Path to tracker results.",
        default='./eval_data/eval_out')
    return arg_parser.parse_args()
####+++++*****  python filter_track_gt_csv.py --video_path_in /resources/michael/videos/Bosch/Neckartor2/DJI_0744-0747.MP4 --csv_path_in /input/dataset/tracker_input/csv/01_Track.csv --track_res_in /input/dataset/filter_in/track_res_in.csv --gt_filtered_out /workspace --track_filtered_out /workspace
# --tracker_fol=/input/trackers --seq_ini=/input/gt/seqinfo.ini --seqmap_file=/input/gt/MOT17-train.txt
##python tracker_app.py --video_path_in /resources/michael/videos/Bosch/Neckartor2/DJI_0744-0747.MP4


if __name__ == '__main__':

    args = create_filter_arg_parser()
    ds = CSVReader(args.gt_path_in, args.track_res_in, args.gt_filtered_out, args.track_filtered_out)

    ds.write_filter_gt_result()

    #ds.write_filter_track_result()
"""""""""" gt_to_rm = np.where(np.logical_and(self.gt_s[:, 0] == fid_list[i], self.gt_s[:, 1] == tid))
                gt_to_rm_next = np.where(np.logical_and(self.gt_s[:, 0] == fid_list[i + 1], self.gt_s[:, 1] == tid))
                # in global match row for (tid,fid) and (fid+1, tid)
                gt_global_match_idx = np.where(
                    np.logical_and(self.gt_fid_match_list == fid_list[i], self.gt_tid_match_list == tid))
                gt_global_match_next_idx = np.where(
                    np.logical_and(self.gt_fid_match_list == fid_list[i + 1], self.gt_tid_match_list == tid))
                gt_s_match_idx = self.gt_global_match_idx[gt_global_match_idx]
                gt_s_match_next_idx = self.gt_global_match_idx[gt_global_match_next_idx]


def get_version1_gt_rm_idx(self):
    remove_gt_idx = []
    tid_unique = np.unique(self.gt_s[:, 1])
    tid_unique = tid_unique.astype(np.int)
    counter = 0
    temporal_tid = []
    tid_rm_possible = []
    t = []
    for tid in tid_unique:
        counter = counter + 1
        print("get gt rm index process: ", 100 * (counter / int(np.shape(tid_unique)[0])), "%", flush=True)
        gt_same_tid = self.gt_s[self.gt_s[:, 1] == tid]
        fid_list = gt_same_tid[:, 0]  # all fid for one tid

        if not (tid in self.rm_gap_tid_dict.keys()):
            gt_suc_row = np.squeeze(np.where(self.gt_s[:, 1] == tid))
            if len(self.rm_gt_idx) == 0:
                self.rm_gt_idx = gt_suc_row
            self.rm_gt_idx = np.hstack((self.rm_gt_idx, gt_suc_row))

        else:  # not gap

            for i in range(len(fid_list) - 1):  # each pair (fid, tid), loop all fids for one tid
                # in all global gt row index for (fid, tid) and (fid_next, tid)
                if tid == 4 and i == 0:
                    st = time.time()
                    t.append(st)
                if tid == 8 and i == 0:
                    et = time.time()
                    t.append(et)
                    print("tid4 f0 to f last, time is", t[1] - t[0])

                if fid_list[i + 1] == np.amax(fid_list, axis=0):  # tid in last fid, endet
                    tid_rm_possible = np.unique(tid_rm_possible)
                    temporal_tid = np.unique(temporal_tid)
                    if tid in self.rm_gap_tid_dict.keys():  # gap tids
                        gt_global_match_next_idx = np.where(
                            np.logical_and(self.gt_fid_match_list == fid_list[i + 1],
                                           self.gt_tid_match_list == tid))
                        gt_s_match_next_idx = self.gt_global_match_idx[gt_global_match_next_idx]
                        if np.any(temporal_tid == tid) != 0 and len(
                                gt_s_match_next_idx) != 0:  # matched in last fid
                            tid_rm_possible = np.hstack((tid_rm_possible, tid))

                    else:  # tids with no gap --- change to above merged
                        gt_to_rm_next = np.where(
                            np.logical_and(self.gt_s[:, 0] == fid_list[i + 1], self.gt_s[:, 1] == tid))
                        gt_to_rm_next = np.squeeze(gt_to_rm_next)
                        remove_gt_idx.append(gt_to_rm_next)

                else:  # not last fid, where tid exists
                    if not (
                            tid in self.rm_gap_tid_dict.keys()):  # no gap, successive case---- > changed see above loop1
                        gt_to_rm = np.where(np.logical_and(self.gt_s[:, 0] == fid_list[i], self.gt_s[:, 1] == tid))
                        gt_to_rm_next = np.where(
                            np.logical_and(self.gt_s[:, 0] == fid_list[i + 1], self.gt_s[:, 1] == tid))
                        if (len(remove_gt_idx) != 0 and fid_list[i] == np.amin(fid_list, axis=0)) or len(
                                remove_gt_idx) == 0:
                            gt_to_rm = np.squeeze(gt_to_rm)
                            remove_gt_idx.append(gt_to_rm)
                        gt_to_rm_next = np.squeeze(gt_to_rm_next)
                        remove_gt_idx.append(gt_to_rm_next)

                    else:  # if tid in self.rm_gap_tid_dict.keys()   all gap cases
                        gt_global_match_next_idx = np.where(
                            np.logical_and(self.gt_fid_match_list == fid_list[i + 1],
                                           self.gt_tid_match_list == tid))
                        gt_global_match_idx = np.where(
                            np.logical_and(self.gt_fid_match_list == fid_list[i], self.gt_tid_match_list == tid))
                        gt_s_match_idx = self.gt_global_match_idx[gt_global_match_idx]
                        gt_s_match_next_idx = self.gt_global_match_idx[gt_global_match_next_idx]

                        if fid_list[i + 1] == fid_list[i] + 1:
                            gt_to_rm_next = np.where(
                                np.logical_and(self.gt_s[:, 0] == fid_list[i + 1], self.gt_s[:, 1] == tid))
                            if fid_list[i] == np.amin(fid_list, axis=0):
                                gt_to_rm = np.where(
                                    np.logical_and(self.gt_s[:, 0] == fid_list[i], self.gt_s[:, 1] == tid))
                                remove_gt_idx.append(np.squeeze(gt_to_rm))
                            remove_gt_idx.append(np.squeeze(gt_to_rm_next))

                        if len(gt_s_match_next_idx) != 0 and len(gt_s_match_idx) != 0:  # found in match_list
                            # print("gt_global_match_idx", gt_global_match_idx)
                            """"""if not np.all(self.gt_s[gt_s_match_idx] == self.gt_s[gt_to_rm]):
                                print("Errors of index: gt_global_match_idx using match_idx_global maps to",
                                      self.gt_s[gt_s_match_idx])
                                print(" are different from gt_to_rm", self.gt_s[gt_to_rm])
                                print("fid:", fid_list[i], "tid", tid)""""""
                            # else: gap found, occlusion case, but tid matched in fid and fid+n dont remove
                            tid_rm_possible = np.hstack(
                                (tid_rm_possible, tid))  # matched again (or always matched) should not evaluate
                            if np.any(temporal_tid == tid):
                                temporal_tid = np.delete(temporal_tid, np.where(temporal_tid == tid))

                        elif len(gt_s_match_next_idx) == 0:  # fid matched, fid+1, tid not matched anymore, FN
                            if not (np.any(tid_rm_possible == tid)):
                                temporal_tid = np.hstack((temporal_tid, tid))"""