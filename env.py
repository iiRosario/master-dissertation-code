import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Diretório do main.py
PATH_LOG = os.path.join(BASE_DIR, "logs")

CLASSES = [0, 1, 2]
TRAIN_SIZE_PERCENTAGE = 0.7
TEST_SIZE_PERCENTAGE = 0.2

DATA_AUG_NOISE_FACTOR = 0.5


### LEARNER CONFIGURATION ###
INIT_TRAINING_PERCENTAGE = 0.1
INIT_TRAINING_EPHOCHS = 3

MODELS= os.path.join(BASE_DIR, "models")


## Active Learning Configuration
NUM_CYCLES = 100
NUM_ANNOTATORS = 30

DATA_DIR = "data"
LABELED = "original_data"
ANNOTATIONS_DIR = os.path.join(DATA_DIR, LABELED, "annotations")


INTERVAL_UPPER_LIMIT = 1.0
INTERVAL_DOWN_LIMIT = 0.0
INTERVAL_NORMAL= (0.05, 0.6)
INTERVAL_SPECIALIZED= (0.6, 1)
INTERVAL_NA=(0.05, 0.3)




# PLOTS CONFIGURATION
CLASS_COLORS=['royalblue', 'tomato', 'goldenrod']