from entities.Annotator import Annotator
from env import *
import random


class Committee:
    def __init__(self, annotators= None, seed=-1):
        self.seed = seed
        self.annotators = annotators if annotators is not None else []
        random.seed(seed)

    def random_answer(self, n):
        return [random.choice(CLASSES) for _ in range(n)]
    


    def __repr__(self):
        return (f"\nCommittee:\n"
                f"Number of Annotators: {len(self.annotators)}\n"
                f"List of Specialists per Class: {self.list_num_specialists}\n")
