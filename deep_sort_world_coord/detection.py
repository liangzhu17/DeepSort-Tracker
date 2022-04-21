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
  # bbox_cam, bbox_w_xy3xy1, bbox_w_xy3lh, bbox_w_xyclh, c, rot_alpha
    def __init__(self, cam_tlwh, w_tlwh, w_tlbr, w_xywh, rot_alpha, class_name, feature): # conf removed
        self.w_tlwh = np.asarray(w_tlwh, dtype=np.float)  # xy3, l(x direction), h(y direction)
        self.w_tlbr = np.asarray(w_tlbr, dtype=np.float)  # xy3, xy1
        self.w_xywh = np.asarray(w_xywh, dtype=np.float)  # xyc, l(x direction), h(y direction)
        self.rot_alpha = np.asarray(rot_alpha, dtype=np.float)
        self.cam_tlwh = np.asarray(cam_tlwh, dtype=np.float)

        #self.confidence = float(confidence)
        self.class_name = class_name
        self.feature = np.asarray(feature, dtype=np.float32)

    def get_class(self):
        return self.class_name

    def to_cam_tlbr(self):   # tlwh ---> tlbr
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.cam_tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_cam_xyah(self):   # tlwh ---> xyah
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        # (center_x, center_y, a, w)

        ret = self.cam_tlwh.copy()
        w = ret[2]
        ret[:2] += ret[2:] / 2
        #ret[2] /= ret[3]   original for pedestrian
        ret[2] = ret[3] / ret[2]
        ret[3] = w
        return ret

    def to_w_tlbr(self):   # tlwh ---> tlbr
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.w_tlbr.copy()

        return ret

    """def to_w_rot_coord_tlbr(self):   # tlwh ---> tlbr
        
        ret = self.w_tlbr.copy()
        xc = self.w_xywh[0]
        yc = self.w_xywh[1]
        l = self.w_tlwh[2]
        h = self.w_tlwh[3]

        ret[0] = xc - l/2
        ret[1] = yc - h/2
        ret[2] = xc + l/2
        ret[3] = yc + h / 2
        return ret"""

    """def to_w_rot_coord_tlwh(self):   # tlwh ---> tlbr
        
        ret = self.w_tlbr.copy()
        xc = self.w_xywh[0]
        yc = self.w_xywh[1]
        l = self.w_tlwh[2]
        h = self.w_tlwh[3]

        ret[0] = xc - l/2
        ret[1] = yc - h/2
        ret[2] = l
        ret[3] = h
        return ret"""

    def get_rot_alpha(self):
        """alpha_arr.
        """
        return self.rot_alpha

    def to_w_xyaw(self):   # tlwh ---> xyaw (xyal)
        """Convert bounding box to format `(center x, center y, aspect ratio,
        w)`, where the aspect ratio is `  height / width`.
        """
        # (center_x, center_y, a, w)

        ret = self.w_xywh.copy()
        w = ret[2]
        ret[2] = ret[3] / w   # a = h/w
        ret[3] = w
        return ret

    def to_w_polygon(self):   # tlbr ---> clockwise 3214
        """Convert bounding box to format `(center x, center y, aspect ratio,
        w)`, where the aspect ratio is `  height / width`.
        """
        # (center_x, center_y, a, w)
        ret = self.w_tlbr.copy()

        l = self.w_tlwh[3]

        x3 = ret[0]
        y3 = ret[1]
        x1 = ret[2]
        y1 = ret[3]

        x2 = x3 + l * np.sin(self.rot_alpha)
        y2 = y3 + l * np.cos(self.rot_alpha)

        x4 = x1 - l * np.sin(self.rot_alpha)
        y4 = y1 - l * np.cos(self.rot_alpha)
        return [[x3, y3], [x2, y2], [x1, y1], [x4, y4]]


