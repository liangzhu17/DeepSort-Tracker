# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track
from scipy.optimize import linear_sum_assignment


def track_pos_dist_cost(tracks, detections, track_indices=None, detection_indices=None):
    """A track-position's distance metric.
    Parameters
    ----------
    tracks : List[deep_sort.track.Track]
        A list of tracks.
    detections : List[deep_sort.detection.Detection]
        A list of detections.
    track_indices : Optional[List[int]]
        A list of indices to tracks that should be matched. Defaults to
        all `tracks`.
    detection_indices : Optional[List[int]]
        A list of indices to detections that should be matched. Defaults
        to all `detections`.
    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    # track_indices: [38,10,21,25], detection_indices: [38,10,21,25]
    # cost_matrix    [0, 1, 2, 3]
    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
    for row, track_idx in enumerate(track_indices):
        txy = np.asarray(tracks[track_idx].mean[0], tracks[track_idx].mean[1])
        candidates = np.asarray([[detections[i].to_xyah()[0], detections[i].to_xyah()[1]] for i in detection_indices])
        d = txy - candidates
        cost_matrix[row, :] = np.sqrt(np.square(d[:, 0]) + np.square(d[:, 1]))

    return cost_matrix


def _cosine_distance(a, b, data_is_normalized=False):
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1. - np.dot(a, b.T)


def _nn_cosine_distance(x, y):
    distances = _cosine_distance(x, y)
    return distances.min(axis=0)


def xdistance(features, targets):
    cost_matrix = np.zeros((len(targets), len(features)))

    for i in range(len(targets)):
        cost_matrix[i, :] = _nn_cosine_distance([features[i]], features)
    return cost_matrix


def find_occlusion_fid(cost_matrix, fid):  # for one tid, get the frame number when it is occluded
    # note the fid not match real frame num, because of occlusion
    fid = int(fid)
    mean_col = np.mean(cost_matrix, axis=0)

    x1 = mean_col[fid+1]
    x2 = mean_col[fid+2]
    x3 = mean_col[fid+3]
    x4 = mean_col[fid+4]
    r1 = abs(x2 - x1) / x1
    r2 = abs(x3 - x2) / x2
    r3 = abs(x4 - x3) / x3

    if r1 > 1.2:   # 120%   # 7,8   7,9  7,10
        return fid-2   # fid+2 -4
    else:
        if r2 > 1.2:
            return fid-1
        else:
            if r3 > 1.2:
                return fid    # fid+4 -4
            else:
                return []


class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=60, n_init=1):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1
        self.centerxy_targets_dict = {}
        self.dist_targets_dict = {}

    def predict(self, frame_num):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        self.fid = frame_num
        for track in self.tracks:
            #if track.time_since_update == 1:            #if track.time_since_update != 1:

            if (self.fid, track.track_id) in self.dist_targets_dict.keys():
                [velx, vely] = self.dist_targets_dict[(self.fid, track.track_id)]
                track.predict_adap(self.kf, velx, vely)
            if (self.fid, track.track_id) not in self.dist_targets_dict.keys():
                track.predict(self.kf)

            self.centerxy_targets_dict[(self.fid, track.track_id)] = [track.mean[0], track.mean[1]]
            if (self.fid-1, track.track_id) in self.centerxy_targets_dict:
                cxi = self.centerxy_targets_dict[(self.fid-1, track.track_id)][0]
                cyi = self.centerxy_targets_dict[(self.fid-1, track.track_id)][1]
                self.dist_targets_dict[(self.fid, track.track_id)] = [track.mean[0]-cxi, track.mean[1]-cyi]
                #track.dist_to_fid_last = self.dist_targets_dict[(self.fid-1, track.track_id)]
            if self.fid > 20 and self.fid / 30 == 0:  # delete the old position data
                for fid in range(self.fid-30, self.fid-3):
                    if (fid, track.track_id) in self.centerxy_targets_dict.keys():
                        self.centerxy_targets_dict.pop((fid, track.track_id))
                    if (fid-1, track.track_id) in self.dist_targets_dict.keys():
                        self.dist_targets_dict.pop((fid-1, track.track_id))


    def update(self, detections):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            ret = detections[detection_idx].to_tlbr()
            if ret[0]<5 or ret[1]<10 or ret[1]>2150 or ret[0]>3830:
                continue
            else:
                self._initiate_track(detections[detection_idx])

        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks]
        features, targets = [], []

        for track in self.tracks:
            features += track.features
            targets += [track.track_id for _ in track.features]
            #if track.time_since_update == 1 and track.state != 5:
                #track.features = []

        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections):
        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)  # cosine distance
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)
            return cost_matrix

        def maha_metric(kf, tracks, dets, track_indices, detection_indices):
            pos_cost = np.zeros([len(track_indices), len(detection_indices)])
            measurements = np.asarray([dets[i].to_xyah() for i in detection_indices])
            for row, track_idx in enumerate(track_indices):
                # mahalanobis distance --> gating distance
                pos_cost[row, :] = kf.maha_norm_distance(
                    tracks[track_idx].mean, tracks[track_idx].covariance, measurements, only_position=True)
            return pos_cost

        # Split track set into confirmed and unconfirmed tracks.
        total_tracks = [i for i, t in enumerate(self.tracks)]
        pos_in_indices = []
        detection_indices = list(range(len(detections)))
        for idx in detection_indices:
            d = detections[idx]
            if d.pos_in:
                pos_in_indices.append(idx)
        # here the unconfirmed_tracks should also take part in matching cascade
        if self.fid < 10:
            detections_candidates = list(set(detection_indices) - set(k for k in pos_in_indices))
            # Associate confirmed tracks using appearance features.
            matches_a, unmatched_tracks_a, unmatched_detections = \
                linear_assignment.matching_cascade(
                    gated_metric, self.metric.matching_threshold, self.max_age,
                    self.tracks, detections, total_tracks)

            # Associate remaining tracks together with unconfirmed tracks using IOU.
            iou_track_candidates = [
                k for k in unmatched_tracks_a if
                self.tracks[k].time_since_update == 1]
            unmatched_tracks_a = [
                k for k in unmatched_tracks_a if
                self.tracks[k].time_since_update != 1]

            matches_b, unmatched_tracks_b, unmatched_detections = \
                linear_assignment.min_cost_matching(
                    iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                    detections, iou_track_candidates, unmatched_detections)

            matches = matches_a + matches_b
            unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
            print("matches, unmatches_tracks, unmat_det", matches, unmatched_tracks, unmatched_detections)
            return matches, unmatched_tracks, unmatched_detections

        else:  # total_tracks not empty, all frames except for fid0
            # Associate confirmed tracks using appearance features.
            # total tracks is the indices for self.tracks

            detections_candidates = list(set(detection_indices) - set(k for k in pos_in_indices))
            matches_a, unmatched_tracks_a, unmatched_detections = linear_assignment.matching_cascade(gated_metric,
                self.metric.matching_threshold, self.max_age, self.tracks, detections=detections,
                   track_indices=total_tracks, detection_indices=detections_candidates)

            # Associate remaining tracks together with unconfirmed tracks using IOU.
            iou_track_candidates = [
                k for k in unmatched_tracks_a if
                self.tracks[k].time_since_update == 1]  # not covered car
            unmatched_tracks_a = [
                k for k in unmatched_tracks_a if
                self.tracks[k].time_since_update != 1]  # cars in covering

            matches_b, unmatched_tracks_b, unmatched_detections_b = \
                    linear_assignment.iou_min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

            matches = matches_a + matches_b
            unmatched_detections = pos_in_indices + unmatched_detections_b
            unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
            print("Defined position come in and leave: ", "matches, unmatched_tracks, unmatched_detections,",
                  matches, unmatched_tracks, unmatched_detections)
            return matches, unmatched_tracks, unmatched_detections
            #tracks_cand_c = unmatched_tracks_a + unmatched_tracks_b
    """if len(unmatched_detections_b) != 0 and len(tracks_cand_c) != 0:
                # unmatched detections = unmat_det_b
                cost_matrix = linear_assignment.maha_cost(self.kf, self.tracks, detections,
                                        track_indices=tracks_cand_c, detection_indices=unmatched_detections_b)
                re_matches, re_unmatched_tracks, re_unmatched_detections = \
                linear_assignment.maha_min_cost_matching(cost_matrix, tracks_cand_c, unmatched_detections_b)

                matches = matches + re_matches
                unmatched_tracks = list(set(tracks_cand_c) - set(k for k, _ in re_matches))
                unmatched_detections = re_unmatched_detections + pos_in_indices
    print("Match c: matches, unmatches_tracks, unmat_det", matches, unmatched_tracks, unmatched_detections)"""


    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        class_name = detection.get_class()
        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            feature=detection.feature, class_name=class_name))
        self._next_id += 1


"""# here the unconfirmed_tracks should also take part in matching cascade
        if self.fid < 10:
            detections_candidates = list(set(detection_indices) - set(k for k in pos_in_indices))
            # Associate confirmed tracks using appearance features.
            matches_a, unmatched_tracks_a, unmatched_detections = \
                linear_assignment.matching_cascade(
                    gated_metric, self.metric.matching_threshold, self.max_age,
                    self.tracks, detections, total_tracks)

            # Associate remaining tracks together with unconfirmed tracks using IOU.
            iou_track_candidates = [
                k for k in unmatched_tracks_a if
                self.tracks[k].time_since_update == 1]
            unmatched_tracks_a = [
                k for k in unmatched_tracks_a if
                self.tracks[k].time_since_update != 1]

            matches_b, unmatched_tracks_b, unmatched_detections = \
                linear_assignment.min_cost_matching(
                    iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                    detections, iou_track_candidates, unmatched_detections)

            matches = matches_a + matches_b
            unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
            print("matches, unmatches_tracks, unmat_det", matches, unmatched_tracks, unmatched_detections)
            return matches, unmatched_tracks, unmatched_detections

        else:  # total_tracks not empty, all frames except for fid0
            # Associate confirmed tracks using appearance features.
            # total tracks is the indices for self.tracks

            detections_candidates = list(set(detection_indices) - set(k for k in pos_in_indices))
            matches_a, unmatched_tracks_a, unmatched_detections = linear_assignment.matching_cascade(gated_metric,
                self.metric.matching_threshold, self.max_age, self.tracks, detections=detections,
                   track_indices=total_tracks, detection_indices=detections_candidates)

            # Associate remaining tracks together with unconfirmed tracks using IOU.
            iou_track_candidates = [
                k for k in unmatched_tracks_a if
                self.tracks[k].time_since_update == 1]  # not covered car
            unmatched_tracks_a = [
                k for k in unmatched_tracks_a if
                self.tracks[k].time_since_update != 1]  # cars in covering

            matches_b, unmatched_tracks_b, unmatched_detections_b = \
                    linear_assignment.iou_min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

            matches = matches_a + matches_b
            unmatched_detections = pos_in_indices + unmatched_detections_b
            unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
            print("Defined position come in and leave: ", "matches, unmatched_tracks, unmatched_detections,",
                  matches, unmatched_tracks, unmatched_detections)

            tracks_cand_c = unmatched_tracks_a + unmatched_tracks_b
            if len(unmatched_detections_b) != 0 and len(tracks_cand_c) != 0:
                # unmatched detections = unmat_det_b
                cost_matrix = linear_assignment.maha_cost(self.kf, self.tracks, detections,
                                        track_indices=tracks_cand_c, detection_indices=unmatched_detections_b)
                re_matches, re_unmatched_tracks, re_unmatched_detections = \
                linear_assignment.maha_min_cost_matching(cost_matrix, tracks_cand_c, unmatched_detections_b)

                matches = matches + re_matches

                unmatched_tracks = list(set(tracks_cand_c) - set(k for k, _ in re_matches))
                unmatched_detections = re_unmatched_detections + pos_in_indices
                print("Match c: matches, unmatches_tracks, unmat_det", matches, unmatched_tracks, unmatched_detections)"""
            #return matches, unmatched_tracks, unmatched_detections


"""self.centerxy_targets_dict[(self.fid, track.track_id)] = [track.mean[0], track.mean[1]]
            if (self.fid-1, track.track_id) in self.centerxy_targets_dict:
                cxi = self.centerxy_targets_dict[(self.fid-1, track.track_id)][0]
                cyi = self.centerxy_targets_dict[(self.fid-1, track.track_id)][1]
                self.dist_targets_dict[(self.fid-1, track.track_id)] = np.sqrt(
                    np.square(track.mean[0]-cxi) + np.square(track.mean[1]-cyi))
                track.dist_to_fid_last = self.dist_targets_dict[(self.fid-1, track.track_id)]
            if self.fid > 20 and self.fid / 30 == 0:  # delete the old position data
                for fid in range(self.fid-30, self.fid-3):
                    if (fid, track.track_id) in self.centerxy_targets_dict.keys():
                        self.centerxy_targets_dict.pop((fid, track.track_id))
                    if (fid-1, track.track_id) in self.dist_targets_dict.keys():
                        self.dist_targets_dict.pop((fid-1, track.track_id))"""