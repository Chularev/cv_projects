'''
@Article{electronics10030279,
    AUTHOR = {Padilla, Rafael and Passos, Wesley L. and Dias, Thadeu L. B. and Netto, Sergio L. and da Silva, Eduardo A. B.},
    TITLE = {A Comparative Analysis of Object Detection Metrics with a Companion Open-Source Toolkit},
    JOURNAL = {Electronics},
    VOLUME = {10},
    YEAR = {2021},
    NUMBER = {3},
    ARTICLE-NUMBER = {279},
    URL = {https://www.mdpi.com/2079-9292/10/3/279},
    ISSN = {2079-9292},
    DOI = {10.3390/electronics10030279}
}
'''
from torchmetrics import Metric
import torchvision.ops as ops
import torch


class Iou(Metric):
    def __init__(self, dist_sync_on_step=False, threshold=0.75):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.threshold = threshold

        self.add_state("iou", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0, dtype=torch.int32), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        area1 = ops.box_area(preds)
        area2 = ops.box_area(target)

        lt = torch.max(preds[:, :2], target[:, :2])
        rb = torch.min(preds[:, 2:], target[:, 2:])

        wh = (rb - lt).clamp(min=0)
        inter = wh[:, 0] * wh[:, 1]

        union = area1 + area2 - inter

        self.iou += torch.sum(inter / union)
        self.total += preds.shape[0]

    def compute(self):
        return self.iou.float() / self.total


class Iou_old:
    @staticmethod
    def iou(boxA, boxB):
        # if boxes dont intersect
        if Iou._boxesIntersect(boxA, boxB) is False:
            return 0
        interArea = Iou._getIntersectionArea(boxA, boxB)
        union = Iou._getUnionAreas(boxA, boxB, interArea=interArea)
        # intersection over union
        iou = interArea / union
        assert iou >= 0
        return iou

    # boxA = (Ax1,Ay1,Ax2,Ay2)
    # boxB = (Bx1,By1,Bx2,By2)
    @staticmethod
    def _boxesIntersect(boxA, boxB):
        if boxA[0] > boxB[2]:
            return False  # boxA is right of boxB
        if boxB[0] > boxA[2]:
            return False  # boxA is left of boxB
        if boxA[3] < boxB[1]:
            return False  # boxA is above boxB
        if boxA[1] > boxB[3]:
            return False  # boxA is below boxB
        return True

    @staticmethod
    def _getIntersectionArea(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # intersection area
        return (xB - xA + 1) * (yB - yA + 1)

    @staticmethod
    def _getUnionAreas(boxA, boxB, interArea=None):
        area_A = Iou._getArea(boxA)
        area_B = Iou._getArea(boxB)
        if interArea is None:
            interArea = Iou._getIntersectionArea(boxA, boxB)
        return float(area_A + area_B - interArea)

    @staticmethod
    def _getArea(box):
        return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
