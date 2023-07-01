import torch
from core.train_context import _TrainContext
from helpers.bar import Bar


class ValidatorMetric:

    def __init__(self, context: _TrainContext):
        self.c = context
        self.metrics = self.c.metric(self.c.device)
        self.checkpointer = self.c.metric_checkpointer

        self.bar = Bar(self.c.val_loader)
        self.bar.set('phase', 'validation_metric')

    def metric_calc(self, target, model, train_idx):
        img, bboxes = target

        img = img.to(self.c.device, torch.float)
        bboxes = bboxes.to(self.c.device, torch.float)

        prediction = model(img)

        metrics = self.metrics(prediction, bboxes, img.shape[0], train_idx)
        for key in metrics.keys():
            item = metrics[key].item()
            self.bar.set(key, item)
            self.c.logger.add_scalar('Validation/batch/metric/{}'.format(key), item)

        return sum(metrics.values())

    def _loop(self, epoch):
        self.bar.set('epoch', epoch)
        self.bar.start()

        self.c.model.eval()

        metric_accum = 0
        for i_step, target in enumerate(self.bar):
            metric = self.metric_calc(target, self.c.model, i_step)

            metric_accum += metric.item()
            self.c.logger.add_scalar('Validation/batch/metric/sum', metric.item())

            ave_metric = metric_accum / (i_step + 1)
            self.c.logger.add_scalar('Validation/batch/metric/average', ave_metric)

            self.bar.set('ave_metric', ave_metric)
            self.bar.update()

        self.bar.stop()

        metric = self.metrics.compute()
        self.metrics.reset()
        return metric['map'].item()

    @torch.no_grad()
    def step(self, epoch):
        if epoch % self.checkpointer.checkpoint_frequency == 0 and epoch > 0:
            metric = self._loop(epoch)
            self.c.logger.add_scalar('Validation/epoch/metric', metric)

            if self.checkpointer.is_finish(metric):
                return False

            self.checkpointer.save(metric)

        return True
