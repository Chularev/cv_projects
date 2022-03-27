from torchvision.models import resnet34
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = resnet34(pretrained=True)
        self.model.fc = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, 5))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)

        return {
            'class': self.sigmoid(x[:, 0]),
            'bbox': x[:, 1:]
        }
