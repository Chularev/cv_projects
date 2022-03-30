import torch
import numpy as np
import cv2
from PIL import Image
import os
from viewer import Viewer


class RoadDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            path,
            transforms=None
    ):
        self.transforms = transforms

        self.img_paths = os.listdir(path)
        self.path = path
        self.viewer = Viewer()

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img_path = self.path + '/' + img_path

        img = cv2.imread(img_path)

        img_path_target = img_path.replace("image_2", "gt_image_2")
        img_path_target = img_path_target.replace("um_", "um_road_")
        img_path_target = img_path_target.replace("umm_", "umm_road_")
        img_path_target = img_path_target.replace("uu_", "uu_road_")

        img_target = cv2.imread(img_path_target)

        m = self.get_mask()
        mask = np.all(img_target == m['road_label'], axis=2)
        mask = self.viewer.binary(img_target, mask)

        if self.transforms is not None:
            tmp = self.transforms(image=img, mask=mask)
            img = tmp['image']
            mask = tmp['mask']

        return {'image': img, "mask": mask, 'path': self.path + '/' + self.img_paths[idx]}

    def __len__(self):
        return len(self.img_paths)

    def get_mask(self):
        return {
            'non_road_label': np.array([255, 0, 0]),
            'road_label': np.array([255, 0, 255]),
            'other_road_label': np.array([0, 0, 0])
        }