# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from scipy.optimize import linear_sum_assignment
from . import kalman_filter


INFTY_COST = 1e+5
def return_ind_original_nlem(ori, col_rm): # n >= m
    #ori = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])  # aft delete, (track ind) col match result
    #ind = np.array([0,1,2,3,4,5,6])  # aft delete, (track ind) col match result
    #col_rm = np.array([2,4,6])

    a = np.delete(ori, col_rm, axis=0)

    return a

def return_ind_original_nsm(m, col_rm, ind):
    #ori = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])  # aft delete, (track ind) col match result
    #ind = np.array([0,1,2,4,5])  # aft delete, (track ind) col match result
    #col_rm = np.array([2,4,6])
    ori = np.arange(m+1)
    all_match = np.delete(ori, col_rm, axis=0)
    l_s = np.arange(len(all_match))
    l_missed = []
    for i, l in enumerate(l_s):
        if not(np.any(l==ind)):
            l_missed.append(i)

    l_missed = np.asarray(l_missed)
    a = np.delete(all_match, l_missed, axis=0)
    return a


def get_ori_col_index(col_arr,col_rm):
    # range from 0 to 70, shape=70, just no order
    #col_arr = np.asarray([0,1,2,3,4,5,20,21,25,22,23,24,6,7,8,9,10,11,27,28,29,30,31,32,33,34,12,13,14,15,16,
    #                      17,18,19,26,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,52,50,51], dtype=int)
    # col_rm length = 3
    col_new = col_arr.copy()
    # n >= m: column all matched
    if len(col_rm)==1:
        for i, c in enumerate(col_new):
            if c >= col_rm:
                col_new[i]=c+1
    if len(col_rm)==2:   # works for 2,3 or 2,18
        for i, c in enumerate(col_new):
            if col_rm[0] <= c < col_rm[1]-1:
                col_new[i] = c + 1
            if c >= col_rm[1]-1:
                col_new[i] = c + 2
    if len(col_rm)==3:  # 2,3, 18; 2, 4,7; 2,3,4; 2, 18, 19
        for i, c in enumerate(col_new):
            if col_rm[0] <= c < col_rm[1]-1:  #  [2,17)
                col_new[i] = c + 1
            if col_rm[1]-1 <= c < col_rm[2]-2:  # [17, 17)in valid
                col_new[i] = c + 2
            if c >= col_rm[2]-2:    # [16,)
                col_new[i] = c + 3
    return col_new

def min_cost_matching(
        distance_metric, max_distance, tracks, detections, track_indices=None,
        detection_indices=None):
    """Solve linear assignment problem.

    Parameters
    ----------
    distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
        The distance metric is given a list of tracks and detections as well as
        a list of N track indices and M detection indices. The metric should
        return the NxM dimensional cost matrix, where element (i, j) is the
        association cost between the i-th track in the given track indices and
        the j-th detection in the given detection_indices.
    max_distance : float
        Gating threshold. Associations with cost larger than this value are
        disregarded.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : List[int]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above).
    detection_indices : List[int]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above).

    Returns
    -------
    (List[(int, int)], List[int], List[int])
        Returns a tuple with the following three entries:
        * A list of matched track and detection indices.
        * A list of unmatched track indices.
        * A list of unmatched detection indices.

    """
    if track_indices is None:    # generate sequential numbers as index
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    if len(detection_indices) == 0 or len(track_indices) == 0:
        return [], track_indices, detection_indices  # Nothing to match.

    cost_matrix = distance_metric(
        tracks, detections, track_indices, detection_indices)
    cost_matrix[cost_matrix > max_distance] = 999 + 1e-5
    det_len = np.shape(cost_matrix)[1]
    col_rm = np.asarray([], dtype=int)

    for c in range(det_len):
        if np.all(cost_matrix[:, c] == 999 + 1e-5):
            col_rm = np.hstack((col_rm, c))  # col numbers

    col_rm = col_rm.astype(int)

    if np.shape(col_rm)==(0,):

        indices = linear_sum_assignment(cost_matrix)
        indices = np.asarray(indices)
        indices = np.transpose(indices)
        matches, unmatched_tracks, unmatched_detections = [], [], []
        for col, detection_idx in enumerate(detection_indices):
            if col not in indices[:, 1]:
                unmatched_detections.append(detection_idx)
        for row, track_idx in enumerate(track_indices):
            if row not in indices[:, 0]:
                unmatched_tracks.append(track_idx)
        for row, col in indices:
            track_idx = track_indices[row]
            detection_idx = detection_indices[col]
            if cost_matrix[row, col] > max_distance:
                unmatched_tracks.append(track_idx)
                unmatched_detections.append(detection_idx)
            else:
                matches.append((track_idx, detection_idx))
        return matches, unmatched_tracks, unmatched_detections


    else:
        cost_new = np.delete(cost_matrix, col_rm,
                             axis=1)  # no invalid col, which should not be matched, row numbers not changed
        n = np.shape(cost_new)[0]
        m = np.shape(cost_new)[1]
        del_indices = linear_sum_assignment(cost_new)
        del_indices = np.asarray(del_indices)
        del_indices = np.transpose(del_indices)  # to -- > (row_ind, col_ind) form  (track_ind, det_ind)
        ori_col_ind = np.asarray([], dtype=int)
        if n >= m:
            ori_col_ind = get_ori_col_index(del_indices[:, 1], col_rm)
        if n < m:
            ori_col_ind = return_ind_original_nsm(m, col_rm, del_indices[:, 1])

        if np.shape(ori_col_ind) != np.shape(np.unique(ori_col_ind)):
            print("Error in cost matrix: Same reID cost from two detections")

        matches, unmatched_tracks, unmatched_detections = [], [], []
        for col, detection_idx in enumerate(detection_indices):
            if col not in ori_col_ind:
                unmatched_detections.append(detection_idx)
        for row, track_idx in enumerate(track_indices):
            if row not in del_indices[:, 0]:
                unmatched_tracks.append(track_idx)

        for i, col in enumerate(ori_col_ind):
            row = del_indices[:, 0][i]
            track_idx = int(track_indices[row])
            detection_idx = detection_indices[col]

            if cost_matrix[row, col] > max_distance:
                unmatched_tracks.append(track_idx)
                unmatched_detections.append(detection_idx)
            else:
                matches.append((track_idx, detection_idx))
        return matches, unmatched_tracks, unmatched_detections


def iou_min_cost_matching(
        distance_metric, max_distance, tracks, detections, track_indices=None,
        detection_indices=None):
    """Solve linear assignment problem.

    Parameters
    ----------
    distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
        The distance metric is given a list of tracks and detections as well as
        a list of N track indices and M detection indices. The metric should
        return the NxM dimensional cost matrix, where element (i, j) is the
        association cost between the i-th track in the given track indices and
        the j-th detection in the given detection_indices.
    max_distance : float
        Gating threshold. Associations with cost larger than this value are
        disregarded.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : List[int]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above).
    detection_indices : List[int]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above).

    Returns
    -------
    (List[(int, int)], List[int], List[int])
        Returns a tuple with the following three entries:
        * A list of matched track and detection indices.
        * A list of unmatched track indices.
        * A list of unmatched detection indices.

    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    if len(detection_indices) == 0 or len(track_indices) == 0:
        return [], track_indices, detection_indices  # Nothing to match.

    cost_matrix = distance_metric(
        tracks, detections, track_indices, detection_indices)
    cost_matrix[cost_matrix > max_distance] = 9e+3 + 1e-5
    indices = linear_sum_assignment(cost_matrix)
    indices = np.asarray(indices)
    indices = np.transpose(indices)
    matches, unmatched_tracks, unmatched_detections = [], [], []
    for col, detection_idx in enumerate(detection_indices):
        if col not in indices[:, 1]:
            unmatched_detections.append(detection_idx)
    for row, track_idx in enumerate(track_indices):
        if row not in indices[:, 0]:
            unmatched_tracks.append(track_idx)
    for row, col in indices:
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        if cost_matrix[row, col] > max_distance:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx, detection_idx))
    return matches, unmatched_tracks, unmatched_detections


def matching_cascade(
        distance_metric, max_distance, cascade_depth, tracks, detections,
        track_indices=None, detection_indices=None):
    """Run matching cascade.

    Parameters
    ----------
    distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
        The distance metric is given a list of tracks and detections as well as
        a list of N track indices and M detection indices. The metric should
        return the NxM dimensional cost matrix, where element (i, j) is the
        association cost between the i-th track in the given track indices and
        the j-th detection in the given detection indices.
    max_distance : float
        Gating threshold. Associations with cost larger than this value are
        disregarded.
    cascade_depth: int
        The cascade depth, should be se to the maximum track age.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : Optional[List[int]]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above). Defaults to all tracks.
    detection_indices : Optional[List[int]]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above). Defaults to all
        detections.

    Returns
    -------
    (List[(int, int)], List[int], List[int])
        Returns a tuple with the following three entries:
        * A list of matched track and detection indices.
        * A list of unmatched track indices.
        * A list of unmatched detection indices.

    """
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))

    unmatched_detections = detection_indices
    matches = []
    for level in range(cascade_depth):
        if len(unmatched_detections) == 0:  # No detections left
            break

        track_indices_l = [
            k for k in track_indices
            if tracks[k].time_since_update == 1 + level
        ]
        if len(track_indices_l) == 0:  # Nothing to match at this level
            continue

        matches_l, _, unmatched_detections = \
            min_cost_matching(
                distance_metric, max_distance, tracks, detections,
                track_indices_l, unmatched_detections)
        matches += matches_l
    unmatched_tracks = list(set(track_indices) - set(k for k, _ in matches))
    return matches, unmatched_tracks, unmatched_detections


def gate_cost_matrix(
        kf, cost_matrix, tracks, detections, track_indices, detection_indices,
        gated_cost=INFTY_COST, only_position=False):
    """Invalidate infeasible entries in cost matrix based on the state
    distributions obtained by Kalman filtering.

    Parameters
    ----------
    kf : The Kalman filter.
    cost_matrix : ndarray
        The NxM dimensional cost matrix, where N is the number of track indices
        and M is the number of detection indices, such that entry (i, j) is the
        association cost between `tracks[track_indices[i]]` and
        `detections[detection_indices[j]]`.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : List[int]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above).
    detection_indices : List[int]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above).
    gated_cost : Optional[float]
        Entries in the cost matrix corresponding to infeasible associations are
        set this value. Defaults to a very large value.
    only_position : Optional[bool]
        If True, only the x, y position of the state distribution is considered
        during gating. Defaults to False.

    Returns
    -------
    ndarray
        Returns the modified cost matrix.

    """
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray(
        [detections[i].to_w_xyaw() for i in detection_indices])
    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = gated_cost
    return cost_matrix
