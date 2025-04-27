import numpy as np
from env import *

class Annotator:
    def __init__(self, id=-1, seed=0, num_classes=10):
        """
        Class representing an annotator.
        
        :param id_: Unique identifier for the annotator.
        :param alphas: List of alpha values (if not provided, it will be initialized with random values).
        :param specialized_class: Index of the class that the annotator specializes in (single integer).
        :param seed: Seed for the random number generator.
        """
        self.id = id
        self.seed = seed
        self.num_classes = num_classes
        np.random.seed(self.seed)
        self.confusion_matrix = self.init_confusion_matrix()
        self.avg_reputation = 0.00
        self.reputations = [0.00 for _ in range(num_classes)]
        

    def init_confusion_matrix(self):
        """
        Initializes the confusion matrix for the annotator using Dirichlet distributions.

        Each row i represents the probability distribution over the labels assigned
        by the annotator when the true class is i.

        :return: Confusion matrix as a numpy array of shape (num_classes, num_classes)
        """
        confusion_matrix = np.zeros((self.num_classes, self.num_classes))

        for true_class in range(self.num_classes):
            # Build alpha vector for Dirichlet: high value for the correct class, low for others
            alpha_vector = [1.0] * self.num_classes
            alpha_vector[true_class] = self.alphas[true_class]  # Higher confidence in the true class

            # Sample from Dirichlet to get the probability distribution
            confusion_matrix[true_class] = np.random.dirichlet(alpha_vector)

        return confusion_matrix


    def __repr__(self):
        matrix_str = np.array2string(self.confusion_matrix, precision=2, suppress_small=True)
        return (f"\nAnnotator:\n"
                f"ID: {self.id}\n"
                f"Seed: {self.seed}\n"
                f"Reputations: {self.reputations}\n"
                f"Avg Reputation: {self.avg_reputation}\n"
                f"Confusion Matrix:\n{matrix_str}\n")





    def rate_other(self, other):
        """
        Rates another annotator based on their alphas and the specialized class.

        :param other: Another Annotator instance.
        :return: Rating value.
        """
       