from entities.Annotator import Annotator
from env import *
import random


class Committee:
    def __init__(self, size=30, seed=-1):
        self.seed = seed
        self.annotators = []
        random.seed(seed)

        self.labeled_samples_class = [0 for _ in range(len(CLASSES))]



        for i in range(30):
            ann = Annotator(id=i, seed=seed+i, num_classes=len(CLASSES), alphas=None)
            self.annotators.append(ann)



    def random_answer(self, n):
        return [random.choice(CLASSES) for _ in range(n)]
    
    def majority_voting_answer(self, true_target):
        return

    def reputation_based_answer(self, true_target):
        return 




    def __repr__(self):
        return (f"\nCommittee:\n"
                f"Number of Annotators: {len(self.annotators)}\n"
                f"List of Specialists per Class: {self.list_num_specialists}\n")
