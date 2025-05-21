import os
from modAL.uncertainty import uncertainty_sampling, margin_sampling, entropy_sampling
from utils.utils import *

CLASS_COLORS = ['royalblue', 'tomato', 'goldenrod','mediumseagreen', 'orchid', 'slateblue',  'darkorange', 'turquoise', 'firebrick', 'deeppink']

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Diretório do main.py
PATH_LOG = os.path.join(BASE_DIR, "logs")
RESULTS_PATH = os.path.join(BASE_DIR, "runs")
MODELS= os.path.join(BASE_DIR, "models")


TRAIN_SIZE_PERCENTAGE = 0.7
TEST_SIZE_PERCENTAGE = 0.2
VAL_SIZE_PERCENTAGE = 0.1

DATA_AUG_NOISE_FACTOR = 0.5


### LEARNER CONFIGURATION ###
INIT_TRAINING_PERCENTAGE = 0.003
LEARNING_RATE = 0.002
EPHOCS = 10
BATCH_SIZE = 256

NUM_CYCLES = 1
POOL_SIZE = 16
CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

## Active Learning Configuration

ORACLE_SIZE_LARGE = 30
ORACLE_SIZE_MEDIUM = 15
ORACLE_SIZE_SMALL = 5
ORACLE_SIZE_MINI = 2

ORACLE_SIZES = [ORACLE_SIZE_SMALL, ORACLE_SIZE_MEDIUM, ORACLE_SIZE_LARGE]

ORACLE_ANSWER_RANDOM = "random"
ORACLE_ANSWER_REPUTATION = "reputation_based"
ORACLE_ANSWER_GROUND_TRUTH = "ground_truth"
ORACLE_ANSWERS = [ORACLE_ANSWER_REPUTATION, ORACLE_ANSWER_RANDOM, ORACLE_ANSWER_GROUND_TRUTH]

DATASET_CIFAR_10 = "CIFAR-10"
DATASET_MNIST = "MNIST"
DATASET_MNIST_FASHION = "MNIST_FASHION"
DATASET_CIFAR_100 = "CIFAR-100"
DATASET_EMNIST_LETTERS = 'EMNIST_LETTERS'
DATASET_TINY_IMAGENET = 'TINY_IMAGENET'

DATASETS = [DATASET_CIFAR_10, DATASET_MNIST, DATASET_MNIST_FASHION, DATASET_EMNIST_LETTERS]

UNCERTAINTY_SAMPLING = uncertainty_sampling 
MARGIN_SAMPLING = margin_sampling  
ENTROPY_SAMPLING = entropy_sampling

QUERY_STRATEGIES = [UNCERTAINTY_SAMPLING, MARGIN_SAMPLING, ENTROPY_SAMPLING]


# ANNOTATORS
POSITIVE_RATING = 1
NEGATIVE_RATING = 0


WITH_RATING = "with_rating"
WITHOUT_RATING = "without_rating"
WITH_ONLY_RATING = "with_only_rating"

#RATINGS_PERMUTATIONS = [WITHOUT_RATING, WITH_RATING, WITH_ONLY_RATING]
RATINGS_PERMUTATIONS = [WITH_ONLY_RATING]


RANDOM_EXPERTISE = "R"
HIGH_EXPERTISE = "H"
MEDIUM_EXPERTISE = "M"
LOW_EXPERTISE = "L"

EXPERTISES = [LOW_EXPERTISE, MEDIUM_EXPERTISE, HIGH_EXPERTISE ]
#EXPERTISES = [RANDOM_EXPERTISE]




TIN_IMAGENET_ZIP = os.path.join(BASE_DIR, "data", "tiny-imagenet-200.zip")
TIN_IMAGENET_DIR = os.path.join(BASE_DIR, "data", "tiny-imagenet-200")



CLASSES_MNIST = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
CLASSES_MNIST_FASHION = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
CLASSES_EMNIST_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
CLASSES_CIFAR_10 = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]