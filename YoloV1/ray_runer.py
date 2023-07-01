from core.train_starter import start_train
from core.train_parameters import TrainParameters
from core.train_param_enums import LoadStrategy, SaveStrategy
from ray.tune import CLIReporter
from ray import tune
from dataset.dataset_helperr import DatasetHelper

def get_parameters():
    p = TrainParameters()


    p.load_strategy = LoadStrategy.MODEL_OPTIMIZER
    p.save_strategy = SaveStrategy.BEST_MODEL_OPTIMIZER
    p.checkpoint_frequency = 3
    return p



if __name__ == "__main__":

    datasets = DatasetHelper.get_datasets()

    reporter = CLIReporter(max_report_frequency=30)
    analysis = tune.run(
        tune.with_parameters(start_train, datasets=datasets),
        sync_config=tune.SyncConfig(
            syncer=None  # Disable syncing
        ),
        name="YoloV1",
        num_samples=1,
        config={'params': get_parameters()},
        resources_per_trial={"cpu": 3, "gpu": 1},
        progress_reporter=reporter
    )

