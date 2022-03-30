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

        '''
        self.metrics = {
            'train': MyMetric(self.device),
            'val': MyMetric(self.device)
        }
        '''

        self.phase = 'train'
        self.viewer = Viewer()
        self.out = ''

    def to_gpu(self, item):
        return item.type(torch.cuda.FloatTensor).to(self.device)

    def loss_calc(self, target, model):
        gpu_img = self.to_gpu(target['image'])
        gpu_mask = self.to_gpu(target['mask'])

        prediction = model(gpu_img)

        losses = self.losses.calc(prediction, gpu_mask)
        for key in losses.keys():
            self.logger.add_scalar('Losses_{}/{}'.format(self.phase, key), losses[key].item())
            self.out += ' {} - {}'.format(key, losses[key].item())

        '''
        with torch.inference_mode():
            metrics = self.metrics[self.phase].step(prediction, gpu_img_has_person, gpu_box)
            for key in metrics.keys():
                self.logger.add_scalar('Metric_{}/{}'.format(self.phase, key), metrics[key].item())
                self.out += ' {} - {}'.format(key, metrics[key].item())
        '''

        return sum(losses.values())

    def train(self, model, loaders, optimizer, num_epochs, scheduler=None):

        torch.cuda.empty_cache()

        model = model.to(self.device)

        '''
        report_metrics = {
            'loss': {
                'train': [],
                'val': []
            }
        }
        '''

        for epoch in range(num_epochs):
            for phase in ['train'] :#, 'val']:
                if phase == 'train' and epoch > 0:
                    if scheduler is not None:
                        scheduler.step()

                self.phase = phase
                model.train(phase == 'train')  # Set model to training mode

                loss_accum = 0
                step_count = len(loaders[phase])
                for i_step, target in enumerate(loaders[phase]):

                    optimizer.zero_grad()

                    torch.set_grad_enabled(phase == 'train')
                    loss_value = self.loss_calc(target, model)

                    loss_accum += loss_value.item()
                    self.logger.add_scalar('Loss_sum_{}/batch'.format(phase), loss_value.item())
                   # report_metrics['loss'][phase].append(loss_value.item())
                    print('Epoch {}/{}. Phase {} Step {}/{} Loss {}'.format(epoch, num_epochs - 1, phase,
                                                                        i_step, step_count, loss_value.item()) + self.out)
                    self.out = ''

                    if phase == 'train':
                        loss_value.backward()
                        optimizer.step()

                model.train(False)
                torch.set_grad_enabled(False)
                with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    torch.save((model.state_dict(), optimizer.state_dict()), path)

                    for index in range(100, 120):
                        target = self.datasets[phase][index]
                        predict = model(self.to_gpu(target['image'].unsqueeze(0)))
                        predict = predict[0].to('cpu')

                        img_grid = self.viewer.create_output(target, predict)
                        self.logger.add_grid_images('Output ' + str(index), img_grid)

                ave_loss = loss_accum / step_count
                self.logger.add_scalar('Loss_sum_train/epoch', ave_loss)

       # train_metrics = self.metrics['train'].compute()
      #  val_metrics = self.metrics['val'].compute()

'''
        tune.report(
            train_loss=sum(report_metrics['loss']['train']) / len(report_metrics['loss']['train']),
            val_loss=sum(report_metrics['loss']['val']) / len(report_metrics['loss']['val']),

       #     train_iou=train_metrics['iou'].item(),
       #     val_iou=val_metrics['iou'].item(),

        #    train_accuracy=train_metrics['accuracy'].item(),
        #    val_accuracy=val_metrics['accuracy'].item()
        )
'''