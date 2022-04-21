# vim: expandtab:ts=4:sw=4
import numpy as np

class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3
    To_Confirm = 4


class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.

    """

    def __init__(self, mean, covariance, track_id, n_init, max_age, alpha, feature=None, class_name=None):

        self.mean = mean  # xyaw
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        self.state = TrackState.Tentative
        self.features = []
        if feature is not None:
            self.features.append(feature)

        self._n_init = n_init
        self._max_age = max_age
        self.alpha_converted = alpha
        self.class_name = class_name

    def to_w_xywh(self):  # xyaw ---> xywh
        ret = self.mean[:4].copy()
        w = ret[3]
        h = ret[2] * w
        ret[2] = w
        ret[3] = h
        return ret

    def to_w_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.mean[:4].copy()  # xc, yc, a, w

        l = ret[3]
        h = ret[2] * l
        x3 = ret[0] - 0.5 * h * np.sin(self.alpha_converted) - 0.5 * l * np.cos(self.alpha_converted)   # detection x3
        y3 = ret[1] - 0.5 * h * np.cos(self.alpha_converted) + 0.5 * l * np.sin(self.alpha_converted)   # detection y3
        ret[0] = x3
        ret[1] = y3
        ret[2] = l
        ret[3] = h
        return ret

    def to_w_tlbr(self):
        """Get current position in bounding box format `(min x, miny, max x,
        max y)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.mean[:4].copy()  # xc, yc, a, w

        l = ret[3]
        h = ret[2] * l
        x3 = ret[0] - 0.5 * h * np.sin(self.alpha_converted) - 0.5 * l * np.cos(self.alpha_converted)  # detection x3
        y3 = ret[1] - 0.5 * h * np.cos(self.alpha_converted) + 0.5 * l * np.sin(self.alpha_converted)  # detection y3

        x1 = ret[0] + 0.5 * h * np.sin(self.alpha_converted) + 0.5 * l * np.cos(self.alpha_converted)  # detection x1
        y1 = ret[1] + 0.5 * h * np.cos(self.alpha_converted) - 0.5 * l * np.sin(self.alpha_converted)  # detection y1

        ret[0] = x3
        ret[1] = y3
        ret[2] = x1
        ret[3] = y1
        return ret

    """def w_xywh_to_parallel_ax_tlwh(self):  # after rotation parallel to xy axes, xy2, xy4
        ret = self.mean[:4].copy()  # xc, yc, a, w
        xc = ret[0]
        yc = ret[1]
        w = ret[3]
        h = ret[2] * w

        ret[0] = xc - w / 2  # detection xmin
        ret[1] = yc - h / 2  # detection ymin (tl_after rot)
        ret[2] = w
        ret[3] = h
        return ret"""

    """def w_xywh_to_parallel_ax_tlbr(self):  # after rotation parallel to xy axes, xy2, xy4
        ret = self.mean[:4].copy()  # xc, yc, a, w
        xc = ret[0]
        yc = ret[1]
        w = ret[3]
        h = ret[2] * w

        ret[0] = xc - w / 2  # detection x2
        ret[1] = yc - h / 2  # detection y2 (tl_after rot)
        ret[2] = xc + w / 2
        ret[3] = yc + h / 2  # detection y4 (br_after rot)
        return ret"""

    def to_w_polygon(self):  # clockwise 3214
        """Get 4 Polygon point to calculate IoU`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.mean[:4].copy()  # xc, yc, a, w

        l = ret[3]
        h = ret[2] * l
        x3 = ret[0] - 0.5 * h * np.sin(self.alpha_converted) - 0.5 * l * np.cos(self.alpha_converted)  # detection x3
        y3 = ret[1] - 0.5 * h * np.cos(self.alpha_converted) + 0.5 * l * np.sin(self.alpha_converted)  # detection y3

        x1 = ret[0] + 0.5 * h * np.sin(self.alpha_converted) + 0.5 * l * np.cos(self.alpha_converted)  # detection x1
        y1 = ret[1] + 0.5 * h * np.cos(self.alpha_converted) - 0.5 * l * np.sin(self.alpha_converted)  # detection y1

        x2 = x3 + h * np.sin(self.alpha_converted)
        y2 = y3 + h * np.cos(self.alpha_converted)
        x4 = x1 - h * np.sin(self.alpha_converted)
        y4 = y1 - h * np.cos(self.alpha_converted)
        return [[x3, y3], [x2, y2], [x1, y1], [x4, y4]]

    def get_class(self):
        return self.class_name

    def predict(self, kf):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.

        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection
            The associated detection.

        """
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_w_xyaw())
        self.features.append(detection.feature)

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def update_after_initiate(self):   #  direct after initiated, all tracks will stay in view for time until leave
        if self.state == TrackState.Tentative:
            self.state = TrackState.To_Confirm

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        #if self.state == TrackState.Tentative:
            #self.state = TrackState.Deleted
        if self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def to_be_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.To_Confirm

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted
