from data_handlers.dataset import RoadDataset
from data_handlers.augmentations import get_transforms_for_predict
from typing import Dict
'''
    def split(self, validation_split):

        data_size = self.data_handlers.data_handlers.shape[0]
        split = int(np.floor(validation_split * data_size))
        indices = list(range(data_size))
        np.random.shuffle(indices)

        train_indices, val_indices = indices[split:], indices[:split]

        return train_indices, val_indices
'''

def get_datasets() -> Dict[str, RoadDataset]:

    transform = get_transforms_for_predict()

    path = '/mnt/heap/imges/road/training/image_2'
    torch_dataset = RoadDataset(path, transform)

    print('train dataset = ' + str(len(torch_dataset)))

    return {
        'train': torch_dataset
    }