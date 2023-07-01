import torch
from losses.Yolov1 import YoloLoss
from core.train_context import _TrainContext
from helpers.bar import Bar


class Trainer:

    def __init__(self, context: _TrainContext):
        self.c = context
        self.losses = YoloLoss()

        self.bar = Bar(self.c.train_loader)
        self.bar.set('phase', 'train')

    def loss_calc(self, target, model):
        img, bboxes = target

        img = img.to(self.c.device, torch.float)
        bboxes = bboxes.to(self.c.device, torch.float)

        prediction = model(img)

        losses = self.losses(prediction, bboxes)
        for key in losses.keys():
            item = losses[key].item()
            self.bar.set(key, item)
            self.c.logger.add_scalar('Train/batch/loss/{}_train'.format(key), item)

        return sum(losses.values())

    @torch.enable_grad()
    def step(self, epoch):

        self.bar.set('epoch', epoch)
        self.bar.start()

        self.c.model.train()

        if epoch > 0 and self.c.scheduler is not None:
            self.c.scheduler.step()

        loss_accum = 0
        for i_step, target in enumerate(self.bar):
            loss_value = self.loss_calc(target, self.c.model)

            loss_value.backward()
            self.c.optimizer.step()
            self.c.optimizer.zero_grad()

            loss_accum += loss_value.item()
            #self.c.logger.add_scalar('Train/batch/loss/sum_train', loss_value.item())

            ave_loss = loss_accum / (i_step + 1)
            self.c.logger.add_scalar('Train/batch/loss/average_train', ave_loss)

            self.bar.set('ave_loss', ave_loss)
            self.bar.update()

        self.bar.stop()

        self.c.epoch_result['train_loss'] = loss_accum / len(self.c.train_loader)
        return True
