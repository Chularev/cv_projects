from PIL import Image
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
from metrics.metrics_iou import Iou
import io
from torchvision.transforms import ToTensor


class Viewer:
    def convert_from_cv2_to_image(self, img: np.ndarray) -> Image:
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def convert_from_image_to_cv2(self, img: Image) -> np.ndarray:
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    def add_title(self, cv_img, title):
        img = self.convert_from_cv2_to_image(cv_img)
        plt.figure()

        plt.title(title)
        plt.imshow(img)

        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        plt.close()

        img = Image.open(buf)
        return self.convert_from_image_to_cv2(img)

    # Function to calculate mask over image
    def weighted_img(self, img, initial_img, α=1., β=0.5, γ=0.):
        initial_img = np.asarray(initial_img, np.float32)
        img = np.asarray(img, np.float32)
        img[:,:,0] = 0
        img[:, :, 1] = 0
        return cv2.addWeighted(initial_img, α, img, β, γ)

    def mask_image(self, img, mask):
        mask = mask.numpy().astype(np.uint8)
        mask = mask * 255
        newImg = self.weighted_img(mask, img)
        newImg = np.asarray(newImg, np.uint8)
        return newImg

    def binary(self, img, mask):
        newImg = img.copy()
        newImg[:, :, 0] = 1 * mask[:, :]
        newImg[:, :, 1] = 1 * mask[:, :]
        newImg[:, :, 2] = 1 * mask[:, :]
        return newImg

    def prepare_for_grid(self, img, title):
        image = self.convert_from_image_to_cv2(img.copy())
        image = self.add_title(image, title)
        return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)

    def create_output(self, target, prediction = None):
        result = []

        path = target['path']
        img_orig = cv2.imread(path)

        mask = target['mask']

        SIZE = (mask.shape[0], mask.shape[1])
        img_orig = cv2.resize(img_orig, SIZE)

        img = self.prepare_for_grid(img_orig, 'Original img')
        result.append(img)

        mask = self.mask_image(img_orig, mask)
        img = self.prepare_for_grid(mask, 'Mask ')
        result.append(img)

        if prediction != None:
            prediction = torch.round(prediction)
            mask = self.mask_image(img_orig, prediction)
            img = self.prepare_for_grid(mask,'Mask prediction')
            result.append(img)

        return torch.cat(result)


