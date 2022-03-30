from albumentations.pytorch import ToTensorV2
import albumentations as A
import torch
from viewer import Viewer
from logger import Logger
import cv2

def get_a_augmentations():
    return [
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.RandomBrightnessContrast(p=0.3)
    ]
def get_augmentations():
    a_transform = A.Compose(
        get_a_augmentations(),
        bbox_params=A.BboxParams(format='albumentations')
    )
    return a_transform

def get_transforms_for_predict():
    return A.Compose([
        A.Resize(width=128, height=128),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


if __name__ == "__main__":
    #datasets = get_datasets()
    transform = get_a_augmentations()
    logger = Logger('TensorBoard')

    #data = datasets['train']

    #image, target = data[0]

    #image = cv2.imread(target['img_path'])
    image = cv2.imread('/mnt/heap/imges/coco/validation/data/000000161861.jpg')

    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    viewer = Viewer()
    for j in range(4):
        result = []
        image = viewer.add_title(img, 'Original img')
        result.append(torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0))
        for i in range(len(transform) - 1):
            transform[i+1].p = 1
            t2 = transform[i+1]
            tgr = A.Compose([
                transform[0], t2
            ])
            transformed = tgr(image=img)
            image = transformed["image"]
            image = viewer.add_title(image, str(transform[i + 1].__class__.__name__))
            result.append(torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0))

        result = torch.cat(result)
        print('Done inerr {} !'.format(j))
        logger.add_grid_images('my_test', result)

    print('Done All !')
