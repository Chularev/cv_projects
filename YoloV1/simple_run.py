"""
Main file for training Yolo model on Pascal VOC dataset

"""
import os

import torch
from core.train_parameters import TrainParameters
from dataset.dataset_helperr import DatasetHelper
from core.train_starter import start_train


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if not torch.cuda.is_available():
        print("cuda is not available")
        exit(1)

    p = {}
    p['params'] = TrainParameters()
    datasets = DatasetHelper.get_datasets()
    start_train(p, datasets)


if __name__ == "__main__":
    main()
