import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import random
import math
import torch

def split(ratio):
    with open('/mnt/heap/imges/data_birds/images.txt') as f:
        lines = f.read().splitlines()
    class_groups = dict()
    for line in lines:
        value, line = line.split(' ', 1)
        key = line.split('.', 1)[0]
        value = value
        if key in class_groups:
            class_groups[key].append(value)
        else:
            class_groups[key] = [value]

    test_id = []
    for _, group in class_groups.items():
        test_id.extend(random.sample(group, int(math.ceil(len(group) * ratio))))
    train_id = [i for i in map(str, range(1, len(lines) + 1)) if i not in test_id]

    return train_id, test_id

class CUBDataset(Dataset):
    def __init__(self, im_ids, transform=None):
        with open('/mnt/heap/imges/data_birds/images.txt') as f:
            id_to_path = dict([l.split(' ', 1) for l in f.read().splitlines()])
        with open('/mnt/heap/imges/data_birds/bounding_boxes.txt') as f:
            id_to_box = dict()
            for line in f.read().splitlines():
                im_id, *box = line.split(' ')
                id_to_box[im_id] = list(map(float, box))
        self.imgs = [(os.path.join('/mnt/heap/imges/data_birds/images', id_to_path[i]), id_to_box[i])
                     for i in im_ids]
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

    def __getitem__(self, index):
        path, box = self.imgs[index]
        im = Image.open(path).convert('RGB')
        im_size = np.array(im.size, dtype='float32')
        width, height = im_size

        #box = np.array(box, dtype='float32')

        im = self.transform(im)

        target = {}
        target['img_width'] = width
        target['img_height'] = height
        box = [box[0] / width, box[1] / height, (box[0] + box[2]) / width, (box[1] + box[3]) / height]
        target["box"] = torch.as_tensor(box, dtype=torch.float32)

        target["img_has_person"] = True
        target["img_path"] = path


        return im, target

    def __len__(self):
        return len(self.imgs)

