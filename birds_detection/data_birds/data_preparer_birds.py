from data_birds.dataset_birds import split, CUBDataset

def get_datasets():
    train_id, test_id = split(0.2)
    splits = {'train': train_id, 'val': test_id}
    datasets = {split: CUBDataset(splits[split]) for split in ('train', 'val')}
    return datasets