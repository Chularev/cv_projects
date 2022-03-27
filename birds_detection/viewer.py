from PIL import Image
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
from metrics_iou import Iou
import io
from torchvision.transforms import ToTensor


class Viewer:
    def convert_from_cv2_to_image(self, img: np.ndarray) -> Image:
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def convert_from_image_to_cv2(self, img: Image) -> np.ndarray:
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    def convert_box(self, box, width, height):
        return [math.ceil(box[0] * width), math.ceil(box[1] * height), math.ceil(box[2] * width),
                math.ceil(box[3] * height)]

    def add_box(self, img, box, target, color):
        box = self.convert_box(box, target['img_width'], target['img_height'])

        img = self.convert_from_image_to_cv2(img)
        img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]),
                            color, 4)
        return self.convert_from_cv2_to_image(img)

    def box_to_str(self, box):
        result = " ["
        result += '{:.2f}'.format(box[0].item()) + ','
        result += '{:.2f}'.format(box[1].item()) + ','
        result += '{:.2f}'.format(box[2].item()) + ','
        result += '{:.2f}'.format(box[3].item()) + ']'
        return result

    def gen_buffer(self, img, target, prediction):
        plt.figure()
        label = "Gt "
        label += "Yes" if target['img_has_person'] == 1 else 'No'
        if target['img_has_person'] == 1:
            img = self.add_box(img, target['box'], target, (0, 255, 0))

        flag = torch.round(prediction['class'][0]) == 1
        label += "| Pred "
        label += "Yes" if flag else 'No'
        label += ' Prob = {:.2f}'.format(prediction['class'][0])

        if target['box'].numel() > 1:
            iou = Iou()
            tmp = iou(prediction['bbox'], target['box'].unsqueeze(0))
            box = prediction['bbox'][0]
            label += self.box_to_str(box)
            label += ' Iou = {:.2f}'.format(tmp.item())

        if flag:
            img = self.add_box(img, prediction['bbox'][0], target, (0, 0, 255))
        plt.title(label)
        plt.imshow(img)

        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        plt.close()
        return buf

    def get_img_with_predict(self, target, prediction):
        """Create a pyplot plot and save to buffer."""
        img = Image.open(target['img_path']).convert("RGB")
        buffer = self.gen_buffer(img, target, prediction)
        image = Image.open(buffer)
        return ToTensor()(image)

