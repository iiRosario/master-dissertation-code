import numpy as np
from env import *

class Annotator:
    def __init__(self, id=-1, seed=0, num_classes=10, alphas=None):
        """
        Class representing an annotator.

        :param id: Unique identifier for the annotator.
        :param seed: Seed for the random number generator.
        :param num_classes: Number of classes in the classification task.
        :param alphas: List or array of concentration parameters for each class (optional).
                       If None, default alphas are used: high value for correct class, 1.0 for others.
        """
        self.id = id
        self.seed = seed
        self.num_classes = num_classes
        np.random.seed(self.seed)
        
        # If alphas not provided, set a default: higher value for correct label
        if alphas is None:
            self.alphas = [10.0 for _ in range(self.num_classes)]
        else:
            self.alphas = alphas
        
        self.cm_prob = self.init_cm_prob()
        self.cm = np.zeros((self.num_classes, self.num_classes))

        self.reputations = [0.0000 for _ in range(num_classes)]
        self.accuracies = [0.0000 for _ in range(num_classes)]
        self.precisions = [0.0000 for _ in range(num_classes)]
        self.right_labels_class = [0 for _ in range(num_classes)]
        self.wrong_labels_class = [0 for _ in range(num_classes)]

        self.avg_reputation = 0.0000
        self.avg_accuracies = 0.0000
        self.avg_precisions = 0.0000

        self.current_answer = -1


    def init_cm_prob(self):

        cm_prob = np.zeros((self.num_classes, self.num_classes))

        for true_class in range(self.num_classes):
            # Build alpha vector for Dirichlet: high value for the correct class, low for others
            alpha_vector = [1.0] * self.num_classes
            alpha_vector[true_class] = self.alphas[true_class]  # Higher confidence in the true class
            cm_prob[true_class] = np.random.dirichlet(alpha_vector)

        return cm_prob




    def answer(self, true_label):
        probabilities = self.cm_prob[true_label]
        ans = np.random.choice(self.num_classes, p=probabilities)
        self.cm[true_label, ans] += 1     # Update the confusion matrix by incrementing the corresponding cell
        self.current_answer = ans         # Save the answer
        return ans
        


    def update_profile(self):

        return


    def rate(self, other):
        """
        Rates another annotator based on their alphas and the specialized class.

        :param other: Another Annotator instance.
        :return: Rating value.
        """
       

    def __repr__(self):
        matrix_str = np.array2string(self.cm_prob, precision=2, suppress_small=True)
        return (f"Annotator(id={self.id}, "
                f"seed={self.seed}, "
                f"num_classes={self.num_classes}, "
                f"alphas={self.alphas}, "
                f"avg_reputation={self.avg_reputation:.2f})\n"
                f"Confusion Matrix Prob:\n{matrix_str}")