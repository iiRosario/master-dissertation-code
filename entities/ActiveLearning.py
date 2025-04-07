import os
from pycocotools.coco import COCO
from entities.Learner import Learner
from entities.Committee import Committee
from env import *
from utils.DataManager import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Main script directory

class ActiveLearning:
    def __init__(self, committe, learner, limit_classes = 5, num_cycles = 0, seed = 0):
        """
        Class representing the Active Learning process.
        
        :param num_cycles: Number of active learning cycles.
        :param num_classes: Number of classes.
        :param annotators: List of annotators.
        :param learner: List of learner.
        """
        self.limit_classes = limit_classes
        self.annotators = committe.annotators if committe.annotators is not None else []
        self.learner = learner if learner is not None else None
        self.num_cycles = num_cycles
        self.seed = seed
        
        self.coco_train_data = COCO(TRAIN_ANNOTATIONS_FILE)
        self.coco_val_data = COCO(VAL_ANNOTATIONS_FILE)
        self.coco_test_data =  COCO(TEST_ANNOTATIONS_FILE)
        

        classes_train = list_classes(self.coco_train_data, limit=self.limit_classes)
        classes_val = list_classes(self.coco_val_data, limit=self.limit_classes)
        
        if set(classes_train) == set(classes_val): self.classes = classes_train
        else:
            print(f"ERROR AT {self.__class__.__name__}: CLASSES DON'T MATCH")
            exit(0)


        self.train_num_images = count_images(self.coco_train_data)
        self.val_num_images = count_images(self.coco_val_data) 
        self.test_num_images = count_images(self.coco_test_data)
        self.total_num_images =  self.train_num_images + self.val_num_images + self.test_num_images
        self.train_num_annotations = count_annotations(self.coco_train_data) 
        self.val_num_annotations = count_annotations(self.coco_val_data) 
        self.test_num_annotations = count_annotations(self.coco_test_data) 
        self.total_num_annotations = self.train_num_annotations + self.val_num_annotations + self.test_num_annotations

        self.train_imgs = []
        self.validation_imgs = []

        
        

    def __repr__(self):
        return (f"ActiveLearning(\n"
                f"  train_imgs_path='{TRAIN_IMAGES_DIR}',\n"
                f"  test_imgs_path='{TEST_IMAGES_DIR}',\n"
                f"  val_imgs_path='{VAL_IMAGES_DIR}',\n"
                f"  train_annotations_path='{TRAIN_ANNOTATIONS_FILE}',\n"
                f"  test_annotations_path='{TEST_ANNOTATIONS_FILE}',\n"
                f"  val_annotations_path='{VAL_ANNOTATIONS_FILE}',\n"
                f"  ===================  IMAGES LOADED ========================\n"
                f"  train_num_images={self.train_num_images}            {round((self.train_num_images * 100) / self.total_num_images, 2)}%\n"
                f"  test_num_images={self.test_num_images}              {round((self.test_num_images * 100) / self.total_num_images, 2)}%\n"
                f"  val_num_images={self.val_num_images}                {round((self.val_num_images * 100) / self.total_num_images, 2)}%\n"
                f"  ================  ANNOTATIONS LOADED ========================\n"
                f"  train_num_annotations={self.train_num_annotations}        {round((self.train_num_annotations * 100) / self.total_num_annotations, 2)}%\n"
                f"  test_num_annotations={self.test_num_annotations}          {round((self.test_num_annotations * 100) / self.total_num_annotations, 2)}%\n"
                f"  val_num_annotations={self.val_num_annotations}            {round((self.val_num_annotations * 100) / self.total_num_annotations, 2)}%\n"
                f"  ======================== INFO ========================\n"
                f"  num_cycles={self.num_cycles},\n"
                f"  annotators={len(self.annotators)},\n"
                f"  learner={self.learner},\n"
                f"  limit_classes={self.limit_classes},\n"
                f"  classes={self.classes},\n"
                f"  num_classes={len(self.classes)}\n"
                f")")
    