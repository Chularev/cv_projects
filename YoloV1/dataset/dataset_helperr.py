from dataset.vocdataset import VOCDataset
from dataset.augments import Augments as A
from typing import Dict
from helpers.constants import IMG_DIR, LABEL_DIR, TRAIN_CSV_FILE, VAL_CSV_FILE
from helpers.logger import Logger
from helpers.viewer import Viewer


class DatasetHelper:

    @staticmethod
    def get_val_dataset():

        val_dataset = VOCDataset(
            VAL_CSV_FILE, transform=A.validation(), img_dir=IMG_DIR, label_dir=LABEL_DIR,
        )
        print('val dataset = ' + str(len(val_dataset)))
        return val_dataset

    @staticmethod
    def get_train_dataset():
        train_dataset = VOCDataset(
            TRAIN_CSV_FILE,
            transform=A.train(),
            img_dir=IMG_DIR,
            label_dir=LABEL_DIR,
        )

        print('train dataset = ' + str(len(train_dataset)))

        return train_dataset

    @staticmethod
    def get_datasets() -> Dict[str, VOCDataset]:
        return {
            'train': DatasetHelper.get_train_dataset(),
            'val': DatasetHelper.get_val_dataset()
        }


if __name__ == "__main__":
    datasets = DatasetHelper.get_datasets()
    train = datasets['train']

    viewer = Viewer()
    logger = Logger('TensorboardImages')

    for i in range(10):
        target = train[i]

        result = viewer.create_output(target)
        logger.add_grid_images('my_test', result)

        print('Done inerr {} !'.format(i))

    print('Done All !')
