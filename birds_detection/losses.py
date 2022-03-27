from torch import Tensor
from typing import Dict
import torch


class MyLoss:
    def __init__(self) -> None:
        super(MyLoss, self).__init__()

        self.class_loss = torch.nn.BCELoss()
        self.bbox_loss = torch.nn.MSELoss()

    def calc(self, prediction: Tensor, gpu_img_has_person: Tensor, gpu_box: Tensor) -> Dict[str, Tensor]:
        result: Dict[str, Tensor] = {}

        loss_value = self.class_loss(prediction['class'], gpu_img_has_person.type(torch.cuda.FloatTensor))
        result['class_loss'] = loss_value

        indexes_with_label = (gpu_img_has_person == 1).nonzero(as_tuple=True)
        if len(indexes_with_label) > 0:
            result['bbox_loss'] = 100 * self.bbox_loss(prediction['bbox'][indexes_with_label], gpu_box[indexes_with_label])

        return result
