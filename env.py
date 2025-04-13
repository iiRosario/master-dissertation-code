import os
from modAL.uncertainty import uncertainty_sampling, margin_sampling, entropy_sampling

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Diretório do main.py
PATH_LOG = os.path.join(BASE_DIR, "logs")

RESULTS_PATH = os.path.join(BASE_DIR, "runs")



TRAIN_SIZE_PERCENTAGE = 0.7
TEST_SIZE_PERCENTAGE = 0.2

DATA_AUG_NOISE_FACTOR = 0.5


### LEARNER CONFIGURATION ###
INIT_TRAINING_PERCENTAGE = 0.10
INIT_TRAINING_EPHOCHS = 1
INIT_LEARNING_RATE = 0.01

QUERY_STRATEGY = uncertainty_sampling 
#QUERY_STRATEGY = margin_sampling  
#QUERY_STRATEGY = entropy_sampling




MODELS= os.path.join(BASE_DIR, "models")


## Active Learning Configuration
CLASSES = [0, 1, 2, 3]
NUM_CYCLES = 200
NUM_ANNOTATORS = 30

IS_CIFAR = 1


ORACLE_ANSWER_RANDOM = "random"
ORACLE_ANSWER_REPUTATION = "reputation"
ORACLE_ANSWER_GROUND_TRUTH = "ground_truth"

ORACLE_ANSWER = ORACLE_ANSWER_GROUND_TRUTH




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