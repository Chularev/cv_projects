import torch
from pycparser.c_ast import ID

from metrics_iou import Iou
from torchmetrics.classification import Accuracy


class MyMetric:
    def __init__(self, device):
        self.accuracy = Accuracy().to(device)
        self.iou = Iou().to(device)

    def step(self, prediction, gpu_img_has_person, gpu_box):
        return {
            'accuracy': self.accuracy(prediction['class'], gpu_img_has_person),
            'iou': self.iou(prediction['bbox'], gpu_box)
        }

    def compute(self):
        return {
            'accuracy': self.accuracy.compute(),
            'iou': self.iou.compute()
        }