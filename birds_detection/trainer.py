CUDA_LAUNCH_BLOCKING = 1

import numpy as np

import torch
import os
from metrics import MyMetric
from ray import tune
from losses import MyLoss
from logger import Logger
from viewer import Viewer

class Trainer:

    def __init__(self, datasets):
        self.datasets = datasets
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.logger = Logger('TensorBoard')
        self.losses = MyLoss()
        self.metrics = {
            'train': MyMetric(self.device),
            'val': MyMetric(self.device)
        }
        self.phase = 'train'
        self.viewer = Viewer()
        self.out = ''

    def to_gpu(self, item):
        return item.type(torch.cuda.FloatTensor).to(self.device)

    def loss_calc(self, img, target, model):
        gpu_img = self.to_gpu(img)
        gpu_img_has_person = target['img_has_person'].type(torch.cuda.IntTensor).to(self.device)
        gpu_box = self.to_gpu(target['box'])

        prediction = model(gpu_img)

        losses = self.losses.calc(prediction, gpu_img_has_person, gpu_box)
        for key in losses.keys():
            self.logger.add_scalar('Losses_{}/{}'.format(self.phase, key), losses[key].item())
            self.out += ' {} - {}'.format(key, losses[key].item())

        with torch.inference_mode():
            metrics = self.metrics[self.phase].step(prediction, gpu_img_has_person, gpu_box)
            for key in metrics.keys():
                self.logger.add_scalar('Metric_{}/{}'.format(self.phase, key), metrics[key].item())
                self.out += ' {} - {}'.format(key, metrics[key].item())

        return sum(losses.values())

    def train(self, model, loaders, optimizer, num_epochs, scheduler=None):

        torch.cuda.empty_cache()

        model = model.to(self.device)

        report_metrics = {
            'loss': {
                'train': [],
                'val': []
            }
        }

        for epoch in range(num_epochs):
            for phase in ['train', 'val']:
                if phase == 'train' and epoch > 0:
                    if scheduler is not None:
                        scheduler.step()

                self.phase = phase
                model.train(phase == 'train')  # Set model to training mode

                loss_accum = 0
                step_count = len(loaders[phase])
                for i_step, (img, target) in enumerate(loaders[phase]):

                    optimizer.zero_grad()

                    torch.set_grad_enabled(phase == 'train')
                    loss_value = self.loss_calc(img, target, model)

                    loss_accum += loss_value.item()
                    self.logger.add_scalar('Loss_sum_{}/batch'.format(phase), loss_value.item())
                    report_metrics['loss'][phase].append(loss_value.item())
                    print('Epoch {}/{}. Phase {} Step {}/{} Loss {}'.format(epoch, num_epochs - 1, phase,
                                                                        i_step, step_count, loss_value.item()) + self.out)
                    self.out = ''

                    if phase == 'train':
                        loss_value.backward()
                        optimizer.step()

                        
                if phase == 'val':
                    with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
                        path = os.path.join(checkpoint_dir, "checkpoint")
                        torch.save((model.state_dict(), optimizer.state_dict()), path)

                        for index in range(100, 120):
                            img, target = self.datasets['val'][index]
                            predict = model(self.to_gpu(img.unsqueeze(0)))
                            predict['class'] = predict['class'].to('cpu')
                            predict['bbox'] = predict['bbox'].to('cpu')

                            img_with_bbox = self.viewer.get_img_with_predict(target, predict)
                            self.logger.add_image('Image ' + str(index), img_with_bbox)

                ave_loss = loss_accum / step_count
                self.logger.add_scalar('Loss_sum_train/epoch', ave_loss)

        train_metrics = self.metrics['train'].compute()
        val_metrics = self.metrics['val'].compute()

        tune.report(
            train_loss=sum(report_metrics['loss']['train']) / len(report_metrics['loss']['train']),
            val_loss=sum(report_metrics['loss']['val']) / len(report_metrics['loss']['val']),

            train_iou=train_metrics['iou'].item(),
            val_iou=val_metrics['iou'].item(),

            train_accuracy=train_metrics['accuracy'].item(),
            val_accuracy=val_metrics['accuracy'].item()
        )
