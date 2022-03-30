from torch import Tensor
from typing import Dict
import torch


class MyLoss:
    def __init__(self) -> None:
        super(MyLoss, self).__init__()

        self.loss = torch.nn.BCELoss()

    def calc(self, prediction: Tensor, mask: Tensor) -> Dict[str, Tensor]:
        result = self.loss(prediction, mask)
        return {'BCELoss':  result}
