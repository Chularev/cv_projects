from albumentations.pytorch import ToTensorV2
import albumentations as A
import torch
from helpers.viewer import Viewer
from helpers.logger import Logger
from helpers.constants import IMG_AUG_PATH
import cv2
class Augments:

    @staticmethod
    def get_all_augmentations():

        return [
            A.Resize(448, 448),
            A.HorizontalFlip(p=0.5),

            A.Perspective(always_apply=False, p=0.9, scale=(0.05, 0.1), keep_size=1, pad_mode=0, pad_val=(0, 0, 0), mask_pad_val=0, fit_output=0, interpolation=0),
            #A.PiecewiseAffine(always_apply=False, p=0.5, scale=(0.03, 0.05), nb_rows=(4, 4), nb_cols=(4, 4), interpolation=0, mask_interpolation=0, cval=0, cval_mask=0, mode='constant', absolute_scale=0, keypoints_threshold=0.01)

        ]

    @staticmethod
    def train():
        transform = Augments.get_all_augmentations()
        transform.append(A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
        transform.append(ToTensorV2(p=1.0))
        return A.Compose(
            transform,
            bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
        )

    @staticmethod
    def validation():
        return A.Compose([
            A.Resize(width=448, height=448),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(p=1.0),
            ],
            bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
        )

    @staticmethod
    def predict():
        return A.Compose([
            A.Resize(width=448, height=448),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(p=1.0),
            ]
        )



if __name__ == "__main__":
    aug = Augments()
    logger = Logger('Augments')

    image = cv2.imread(IMG_AUG_PATH)

    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = aug.get_all_augmentations()
    viewer = Viewer()
    for j in range(len(transform)):
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
