
from typing import Dict
from helpers.logger import Logger
import torch
import torch.optim as optim

from core.train_context import _TrainContext
from core.train_parameters import TrainParameters
from core.train_loop import Looper
from core.checkpointers.checkpoint_context import _CheckpointContext
from core.checkpointers.base_checkpointer import BaseCheckpointer
from core.train_param_enums import MetricType
from helpers.constants import CHECKPOINT_FOLDER

def get_loaders(datasets, p:TrainParameters) -> Dict[str, torch.utils.data.DataLoader]:
    return {
        'train': torch.utils.data.DataLoader(
            datasets['train'],
            batch_size=p.t_loader_batch_size,
            num_workers=p.t_loader_num_workers,
            pin_memory=p.t_loader_pin_memory,
            shuffle=p.t_loader_shuffle,
            drop_last=p.t_loader_drop_last,
        ),
        'val': torch.utils.data.DataLoader(
            datasets['val'],
            batch_size=p.v_loader_batch_size,
            num_workers=p.v_loader_num_workers,
            pin_memory=p.v_loader_pin_memory,
            shuffle=p.v_loader_shuffle,
            drop_last=p.v_loader_drop_last,
        )
    }


'''
    This function used to work autocode helper
'''
def covert_to_TrainParameters(parameters) -> TrainParameters:
    return parameters['params']

def create_context_from_params(p: TrainParameters, datasets):

    loaders = get_loaders(datasets, p)
    context = _TrainContext()

    context.train_loader = loaders['train']
    context.val_loader = loaders['val']

    context.model = p.model(split_size=7, num_boxes=2, num_classes=20)

    context.optimizer = p.optimizer(
        context.model.parameters(), lr=p.learning_rate, weight_decay=p.weight_decay
    )

    context.scheduler = None
    if p.scheduler:
        context.scheduler = optim.lr_scheduler.StepLR(
            context.optimizer, step_size=p.scheduler_epoch, gamma=p.scheduler_coefficient
        )

    context.metric = p.metric

    context.epoch_num = p.epoch_num

    context.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    context.logger = Logger('TensorBoard')
    return context

def create_loss_checkpointer(p: TrainParameters, train_context: _TrainContext):
    if not p.loss_checkpointer:
        return None
    c_context = _CheckpointContext()

    c_context.save_strategy = p.l_save_strategy
    c_context.load_strategy = p.l_load_strategy
    c_context.checkpoint_frequency = p.l_checkpoint_frequency
    c_context.metric_type = MetricType.LOSS
    c_context.metric_value_stop = p.l_metric_value_stop

    checkpointer = BaseCheckpointer(c_context, train_context)
    return checkpointer

def create_metric_checkpointer(p: TrainParameters, train_context: _TrainContext):
    if not p.metric_checkpointer:
        return None
    c_context = _CheckpointContext()

    c_context.save_strategy = p.m_save_strategy
    c_context.load_strategy = p.m_load_strategy
    c_context.checkpoint_frequency = p.m_checkpoint_frequency
    c_context.metric_type = MetricType.METRIC
    c_context.metric_value_stop = p.m_metric_value_stop

    checkpointer = BaseCheckpointer(c_context, train_context)
    return checkpointer

def start_train(parameters, datasets):
    # define training and validation dataset loaders

    p = covert_to_TrainParameters(parameters)

    torch.manual_seed(p.seed)

    context = create_context_from_params(p, datasets)

    context.type_load_model = p.type_load_model
    context.metric_checkpointer = create_metric_checkpointer(p, context)
    context.loss_checkpointer = create_loss_checkpointer(p, context)

    looper = Looper(context)
    looper.train_loop()