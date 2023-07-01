import os

import cv2
import torch
from models.Yolov1 import Yolov1
from dataset.augments import Augments
from helpers.utils import cellboxes_to_boxes, non_max_suppression,plot_image
from helpers.viewer import Viewer
import matplotlib.pyplot as plt
class Predictor:
    def __init__(self, path):

        checkpoint = torch.load(path)
        torch.set_grad_enabled(False)

        self.viewer = Viewer()

        self.model = Yolov1(split_size=7, num_boxes=2, num_classes=20)
        self.model.load_state_dict(checkpoint['model_state'])

        self.model.eval()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #self.model.to(self.device)

        self.transforms = Augments.predict()

    def predict_img(self, path):

        image = cv2.imread(path)
        img = self.opencv_img(image)


        # Create figure and axes
        fig, ax = plt.subplots(1)
        # Display the image
        ax.imshow(img)
        plt.show()

    def opencv_img(self, image):
        x_data = self.transforms(image=image)
        x_data = x_data['image']
        #x_data = x_data.to(self.device, torch.float)
        x_data = x_data.unsqueeze(0)
        prediction = self.model(x_data)


        bboxes = cellboxes_to_boxes(prediction)
        bboxes = non_max_suppression(bboxes[0], iou_threshold=0.5, threshold=0.4, box_format="midpoint")

        img = self.viewer.convert_from_cv2_to_image(image)
        return plot_image(img, bboxes)


if __name__ == "__main__":

    predictor = Predictor('/home/alex/workspace/experiments/best_model/model_metric.pth.tar')

    img_path = '/home/alex/workspace/projects/study_cv/pytorch_framework/data/data/images/'
    predictor.predict_img(img_path + '2008_006796.jpg')
