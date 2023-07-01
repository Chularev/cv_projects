from pathlib import Path

ROOT_DIR = str( Path(__file__).absolute().parent.parent )

IMG_DIR = ROOT_DIR + "/data/data/images"
LABEL_DIR = ROOT_DIR + "/data/data/labels"
TRAIN_CSV_FILE = ROOT_DIR + "/data/train.csv"
VAL_CSV_FILE = ROOT_DIR + "/data/test.csv"
CHECKPOINT_FOLDER = '/home/alex/workspace/experiments/best_model/'
LOG_DIR = '/home/alex/workspace/experiments/logs/'
IMG_AUG_PATH = '/home/alex/workspace/projects/study_cv/pytorch_framework/data/data/images/001414.jpg'


if __name__ == "__main__":
    print('ROOT_DIR = ', ROOT_DIR)
    print("IMG_DIR = ", IMG_DIR)
    print("LABEL_DIR = ", LABEL_DIR)
    print("TRAIN_CSV_FILE = ", TRAIN_CSV_FILE)
    print("VAL_CSV_FILE = ", VAL_CSV_FILE)
    print('CHECKPOINT_FOLDER = ', CHECKPOINT_FOLDER)
    print('LOG_DIR = ', LOG_DIR)