from helpers.constants import CHECKPOINT_FOLDER


class _CheckpointContext:
    load_strategy = None
    metric_type = None
    save_strategy = None
    metric_value_stop = None
    checkpoint_frequency = None