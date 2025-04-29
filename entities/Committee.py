from entities.Annotator import Annotator
from env import *
import random


class Committee:
    def __init__(self, size=30, seed=-1):
        self.seed = seed    
        random.seed(seed)
        
        self.labeling_iteration = 0
        self.labeled_samples_class = [0 for _ in range(len(CLASSES))]
        self.size = size


        self.annotators = []
        for i in range(size):
            ann = Annotator(id=i, seed=seed+i, num_classes=len(CLASSES), alpha=0.5, beta=0.5)
            self.annotators.append(ann)



    def random_answer(self, n):
        return [random.choice(CLASSES) for _ in range(n)]
    
    def majority_voting_answer(self, true_target):
        return

    def weight_reputation_answer(self, true_target):
        committee_answer = -1
        self.labeling_iteration += 1

        # GENERATE ANSWERS
        for annotator in self.annotators:
            annotator.answer(true_target)
        
        # RATE OTHERS (skip self)
        for who_rates in self.annotators:
            for other in self.annotators:
                if who_rates is not other:
                    who_rates.rate(other)
                    print(f"who_rates: {who_rates.current_answer}   other: {other.current_answer}")
            
        sum_reputations = [0.00 for _ in range(len(CLASSES))]
        
        for annotator in self.annotators:
            annotator.update_reputation_per_class(N=self.size, iteration=self.labeling_iteration)
            #print(annotator.rating_scores)          
            class_answer = annotator.current_answer
            class_reputation = annotator.reputations[class_answer]
            sum_reputations[class_answer] += class_reputation


            
        print(f"sum_reputations: {sum_reputations}")
        committee_answer = sum_reputations.index(max(sum_reputations))
        
        return committee_answer




    def __repr__(self):
        return (f"\nCommittee:\n"
                f"Number of Annotators: {len(self.annotators)}\n"
                f"List of Specialists per Class: {self.list_num_specialists}\n")
