# vim: expandtab:ts=4:sw=4
import numpy as np


class Detection(object):
    """
    This class represents a bounding box detection in a single image.

    Parameters
    ----------
    tlwh : array_like
        Bounding box in format `(t, l, w, h)`.
    confidence : float
        Detector confidence score.
    feature : array_like
        A feature vector that describes the object contained in this image.

    Attributes
    ----------
    tlwh : ndarray
        Bounding box in format `(top left x, top left y, width, height)`.
    confidence : ndarray
        Detector confidence score.
    class_name : ndarray
        Detector class.
    feature : ndarray | NoneType
        A feature vector that describes the object contained in this image.

    """

    def __init__(self, tlwh, class_name, feature, id):  # conf removed
        self.tlwh = np.asarray(tlwh, dtype=np.float)
        # self.confidence = float(confidence)
        self.class_name = class_name
        self.feature = np.asarray(feature, dtype=np.float32)
        self.id = np.asarray(id, dtype=np.int)
        self._pos_judge_in()

    def get_class(self):
        return self.class_name

    def to_tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = self.tlwh.copy()
        w = ret[2]
        h = ret[3]
        ret[0] = ret[0] + w/2
        ret[1] = ret[1] + h / 2
        # ret[2] /= ret[3]   original for pedestrian
        ret[2] = h / w
        ret[3] = w
        return ret

    def _pos_judge_in(self):
        # to tlbr
        ret = self.tlwh.copy()   # --> tlbr
        ret[2:] += ret[:2]
        # show in road left 0,1,2, eg1: 1,8; eg2 0,3... # road right 3839, 3839, 3840, 3838
        if (ret[0]<2 and 1218<ret[1]<1380) or (ret[2]>3838 and 852<ret[3]<980) \
                or (ret[1]<2 and 1696<ret[0]<1935) or (2150<ret[2]<2365 and ret[3]>2158):

            self.pos_in = True
        else:
            self.pos_in = False
