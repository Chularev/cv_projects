import torch
from pycparser.c_ast import ID

from helpers.utils import cellboxes_to_boxes, non_max_suppression, mean_average_precision
from torchmetrics import Metric

class MyMetric(Metric):
    def __init__(self, device):
        super().__init__(dist_sync_on_step=False)
        self.to(device)

        self.mean_avg_prec = 0
        self.count = 0

    def update(self, predictions, labels, batch_size, train_idx):

        iou_threshold = 0.5
        threshold = 0.4
        box_format = "midpoint"

        all_pred_boxes = []
        all_true_boxes = []

        true_bboxes = cellboxes_to_boxes(labels)
        bboxes = cellboxes_to_boxes(predictions)

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            )

            # if batch_idx == 0 and idx == 0:    all_pred_boxes = []
            #     all_true_boxes = []
            #
            #     # make sure model is in eval before get bboxes
            #     model.eval()
            #     train_idx = 0
            #    plot_image(x[idx].permute(1,2,0).to("cpu"), nms_boxes)
            #    print(nms_boxes)

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                # many will get converted to 0 pred
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            mean_avg_prec = mean_average_precision(
                all_pred_boxes, all_true_boxes, iou_threshold=0.5, box_format="midpoint"
            )
            self.mean_avg_prec += mean_avg_prec
            self.count += 1

    def compute(self):
        return { 'map': self.mean_avg_prec / self.count }