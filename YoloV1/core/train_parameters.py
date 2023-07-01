from core.train_param_enums import LoadStrategy, SaveStrategy, MetricType, TypeLoadModel
from metrics.metrics import MyMetric
import torch.optim as optim
from models.Yolov1 import Yolov1

class TrainParameters:
    need_train = True
    seed = 123
    optimizer = optim.Adam
    model = Yolov1
    model_name = 'best_net'
    learning_rate = 2e-5
    scheduler_epoch = 100
    scheduler_coefficient = 0.9
    weight_decay = 0.1
    scheduler = None
    epoch_num = 1000
    metric = MyMetric

    # Checkpoints
    type_load_model = TypeLoadModel.BEST_METRIC
    # Metric
    metric_checkpointer = True
    m_load_strategy = LoadStrategy.MODEL_OPTIMIZER
    m_save_strategy = SaveStrategy.BEST_MODEL
    m_metric_type = MetricType.METRIC
    m_metric_value_stop = 0.9
    m_checkpoint_frequency = 5

    # Loss
    loss_checkpointer = True
    l_load_strategy = LoadStrategy.MODEL_OPTIMIZER
    l_save_strategy = SaveStrategy.BEST_MODEL
    l_metric_type = MetricType.METRIC
    l_metric_value_stop = 0
    l_checkpoint_frequency = 5

    # Loaders
    # Train Loader
    t_loader_batch_size = 16
    t_loader_num_workers = 2
    t_loader_pin_memory = True
    t_loader_shuffle = True
    t_loader_drop_last = True

    # Val loader
    v_loader_batch_size = 16
    v_loader_num_workers = 2
    v_loader_pin_memory = True
    v_loader_shuffle = True
    v_loader_drop_last = True

