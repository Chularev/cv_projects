from data_handlers.data_preparer import get_datasets
from viewer import Viewer
from logger import Logger

if __name__ == "__main__":
    datasets = get_datasets()
    train = datasets['train']

    viewer = Viewer()
    logger = Logger('TensorboardImages')

    for i in range(10):
        target = train[i]

        result = viewer.create_output(target)
        logger.add_grid_images('my_test', result)

        print('Done inerr {} !'.format(i))

    print('Done All !')
