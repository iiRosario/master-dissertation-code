import numpy as np
from env import *

class Annotator:
    def __init__(self, id=-1, alphas=None, specialized_class=None, seed=0, num_classes=10, interval_normal=None, interval_specialized=None, interval_na=None):
        """
        Class representing an annotator.
        
        :param id_: Unique identifier for the annotator.
        :param alphas: List of alpha values (if not provided, it will be initialized with random values).
        :param specialized_class: Index of the class that the annotator specializes in (single integer).
        :param seed: Seed for the random number generator.
        """
        self.id = id
        self.seed = seed
        self.interval_normal = interval_normal
        self.interval_specialized = interval_specialized
        self.interval_na = interval_na
        self.num_classes = num_classes
        np.random.seed(self.seed)
        
        self.specialized_class = specialized_class  # Agora é um único inteiro
        self.alphas = alphas if alphas is not None else self.init_alphas()
        self.avg_reputation = 0.00
        self.reputations = [0.00 for _ in range(num_classes)]
        

    def init_alphas(self):
        """
        Initializes the list of alphas, allowing one class to be specialized.

        :return: List of alphas initialized with specialization.
        """
        # Generate alphas normally within the range for all classes
        alphas = np.random.uniform(self.interval_normal[0], self.interval_normal[1], self.num_classes).tolist()

        # If there is a specialized class, apply the specialized range
        if self.specialized_class is not None and 0 <= self.specialized_class < self.num_classes:
            alphas[self.specialized_class] = np.random.uniform(self.interval_specialized[0], self.interval_specialized[1])
            
            # Ensure the value does not exceed the upper limit
            if alphas[self.specialized_class] > INTERVAL_UPPER_LIMIT:
                alphas[self.specialized_class] = INTERVAL_UPPER_LIMIT
        
        # Round all values to two decimal places
        alphas = [round(alpha, 2) for alpha in alphas]

        # Generate the NA class alpha separately
        na_alpha = round(np.random.uniform(self.interval_na[0], self.interval_na[1]), 2)
        alphas.append(na_alpha)
        
        return alphas

    def add_alpha(self, alpha):
        """Adds a new alpha value to the list."""
        self.alphas.append(alpha)

    def get_alphas(self) -> list:
        """Returns the list of alphas."""
        return self.alphas

    def __repr__(self):
        return (f"\nAnnotator:\n"
                f"ID: {self.id}\n"
                f"Seed: {self.seed}\n"
                f"Specialized Class: {self.specialized_class}\n"
                f"Alphas: {self.alphas}\n"
                f"Reputations: {self.reputations}\n"
                f"Avg Reputation: {self.avg_reputation}\n")
