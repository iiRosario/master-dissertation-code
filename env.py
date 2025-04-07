import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Diretório do main.py
PATH_LOG = os.path.join(BASE_DIR, "logs")

INIT_MODEL_LEARNER = os.path.join(BASE_DIR, "models", "yolov10m.pt" )
PATH_SAVED_MODELS_LEARNER_A = os.path.join(BASE_DIR, "models", "train", "learner_A")
PATH_SAVED_MODELS_LEARNER_B = os.path.join(BASE_DIR, "models", "train", "learner_B")
PATH_SAVED_MODELS_LEARNER_C = os.path.join(BASE_DIR, "models", "train", "learner_C")


DATA_DIR = "data"
LABELED = "labeled"
ANNOTATIONS_DIR = os.path.join(DATA_DIR, LABELED, "annotations")
TRAIN_IMAGES_DIR = os.path.join(DATA_DIR, LABELED, "train")
VAL_IMAGES_DIR = os.path.join(DATA_DIR, LABELED, "val")
TEST_IMAGES_DIR = os.path.join(DATA_DIR, LABELED, "test")
COCO_YAML = os.path.join(DATA_DIR, LABELED, "coco.yaml")


# Ficheiros de anotações
TRAIN_ANNOTATIONS_FILE = os.path.join(ANNOTATIONS_DIR, "coco_train.json")
VAL_ANNOTATIONS_FILE = os.path.join(ANNOTATIONS_DIR, "coco_val.json")
TEST_ANNOTATIONS_FILE = os.path.join(ANNOTATIONS_DIR, "coco_test.json")


NUM_CYCLES = 10
NUM_CLASSES = 5
NUM_ANNOTATORS = 30
MAX_NUM_SPECIALIZED_CLASSES = 1



INTERVAL_UPPER_LIMIT = 1.0
INTERVAL_DOWN_LIMIT = 0.0
INTERVAL_NORMAL= (0.05, 0.6)
INTERVAL_SPECIALIZED= (0.6, 1)
INTERVAL_NA=(0.05, 0.3)