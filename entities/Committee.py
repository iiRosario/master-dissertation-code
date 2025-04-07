from entities.Annotator import Annotator
from env import *



class Committee:
    def __init__(self, annotators: list = None):

        self.annotators = annotators if annotators is not None else []
        self.list_num_specialists = self.count_specialists()


    def count_specialists(self):
        list_num_specialists = [0] * NUM_CLASSES

        for annotator in self.annotators:
            list_num_specialists[annotator.specialized_class] += 1
        return list_num_specialists

    def get_annotator_by_id(self, id_):
        """Returns the annotator with the given ID, or None if not found."""
        for annotator in self.annotators:
            if annotator.id == id_:
                return annotator
        return None



    def __repr__(self):
        return (f"\nCommittee:\n"
                f"Number of Annotators: {len(self.annotators)}\n"
                f"List of Specialists per Class: {self.list_num_specialists}\n")
