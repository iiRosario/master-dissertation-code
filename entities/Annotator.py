import numpy as np
from env import *

class Annotator:
    def __init__(self, id=-1, seed=0, num_classes=10, alpha=0.5, beta=0.5, expertise=HIGH_EXPERTISE):
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
        np.random.seed(self.seed)
        self.num_classes = num_classes
        self.alpha=0.5                  #Score Ratings weight
        self.beta=0.5                   #Accuracy  Weight
        self.expertise = expertise
        
        # If alphas not provided, set a default: higher value for correct label
        if(self.expertise == LOW_EXPERTISE):
            self.cm_prob = self.init_cm_prob(target_accuracy=0.4, scale=20.0)
        elif(self.expertise == MEDIUM_EXPERTISE):
            self.cm_prob = self.init_cm_prob(target_accuracy=0.5, scale=20.0)
        elif(self.expertise == HIGH_EXPERTISE):
            self.cm_prob = self.init_cm_prob(target_accuracy=0.6, scale=20.0)
        elif(self.expertise == VERY_HIGH_EXPERTISE):
            self.cm_prob = self.init_cm_prob(target_accuracy=0.8, scale=20.0)
        elif(self.expertise == EXPERT_EXPERTISE):
            self.cm_prob = self.init_cm_prob_specialist()

        self.cm = np.zeros((self.num_classes, self.num_classes))
        self.cm_ratings = np.zeros((self.num_classes, self.num_classes))

        self.reputations = [0.0000 for _ in range(num_classes)]
        self.rating_scores = [0 for _ in range(num_classes)]

        self.accuracies = [0.0000 for _ in range(num_classes)]
        self.precisions = [0.0000 for _ in range(num_classes)]
        self.recalls = [0.0000 for _ in range(num_classes)]
        self.f1_scores = [0.0000 for _ in range(num_classes)]
        
        self.avg_reputation = 0.0000
        self.avg_accuracy = 0.0000
        self.avg_precision = 0.0000
        self.avg_recall = 0.0000
        self.avg_f1_score = 0.0000

        self.current_answer = -1

        
        



    def init_cm_prob(self, target_accuracy=0.8, scale=20.0):
        """
        Inicializa uma matriz de confusão probabilística em que a classe verdadeira
        tem aproximadamente target_accuracy de chance de ser escolhida.
        
        :param target_accuracy: Probabilidade média desejada de acerto (ex: 0.7 para 70%)
        :param scale: Parâmetro de concentração da Dirichlet (quanto maior, menos variação)
        """
        cm_prob = np.zeros((self.num_classes, self.num_classes))
        for true_class in range(self.num_classes):
            alpha_vector = [(1 - target_accuracy) / (self.num_classes - 1) * scale
                            for _ in range(self.num_classes)]
            alpha_vector[true_class] = target_accuracy * scale

            probs = np.random.dirichlet(alpha_vector)
            probs = np.round(probs, 2)

            # Corrige para garantir que a soma seja exatamente 1.0
            diff = 1.0 - probs.sum()
            probs[np.argmax(probs)] += diff

            cm_prob[true_class] = probs
        return cm_prob
    
    def init_cm_prob_specialist(self, expert_accuracy=0.8, min_other_accuracy=0.1, max_other_accuracy=0.5, scale=10.0):
        """
        Inicializa uma matriz de confusão probabilística em que o anotador tem alta precisão
        em apenas uma classe (escolhida aleatoriamente), e acurácia entre 20%-60% nas outras.

        :param expert_accuracy: Probabilidade média desejada de acerto para a classe especialista.
        :param min_other_accuracy: Acerto mínimo nas outras classes.
        :param max_other_accuracy: Acerto máximo nas outras classes.
        :param scale: Parâmetro de concentração da Dirichlet.
        """
        cm_prob = np.zeros((self.num_classes, self.num_classes))

        # Escolher aleatoriamente a classe especialista
        expert_class = np.random.randint(0, self.num_classes)

        for true_class in range(self.num_classes):
            if true_class == expert_class:
                target_accuracy = expert_accuracy
            else:
                target_accuracy = np.random.uniform(min_other_accuracy, max_other_accuracy)

            alpha_vector = [(1 - target_accuracy) / (self.num_classes - 1) * scale
                            for _ in range(self.num_classes)]
            alpha_vector[true_class] = target_accuracy * scale

            probs = np.random.dirichlet(alpha_vector)
            probs = np.round(probs, 2)

            # Corrigir soma para 1.0
            diff = 1.0 - probs.sum()
            probs[np.argmax(probs)] += diff

            cm_prob[true_class] = probs

        #print(f"Classe especialista do anotador: {expert_class}")
        return cm_prob


    def answer(self, true_label):
        probabilities = self.cm_prob[true_label]
        ans = np.random.choice(self.num_classes, p=probabilities)
        self.cm[true_label, ans] += 1     # Update the confusion matrix by incrementing the corresponding cell
        self.current_answer = ans         # Save the answer
        return ans
        
    def rate(self, other, true_label=None):
        result = 0
        if self.current_answer != other.current_answer:
            result = NEGATIVE_RATING
        else:
            result = POSITIVE_RATING
            print("POSITIVE RATING")
        other.rating_scores[other.current_answer] += result

        self.cm_ratings[true_label, self.current_answer] += 1

        return result

    def rating_score_value(self, score_i, n_annotators, labeling_iteration):
        return (self.rating_scores[score_i] / (n_annotators*labeling_iteration))


    def update_reputation_per_class(self, N, iteration):
        self.update_accuracy()

        for i in range(len(self.reputations)):
            # R = Ratingscore * alpha + Accuracy * beta 
            rating = self.rating_score_value(i, N, iteration)
            accuracy = self.accuracies[i]
            #print(f"ID: {self.id} Class: {i} Rating: {rating}, Accuracy: {accuracy}")
            self.reputations[i] = rating * self.alpha + accuracy * self.beta
        
        return self.reputations




    def update_accuracy(self):
        self.accuracies = self.accuracy_class()
        self.avg_accuracy = np.mean(self.accuracies)
        return

    def update_all_metrics(self):
        self.accuracies = self.accuracy_class()
        self.precisions = self.precision_class()
        self.recalls = self.recall_class()
        self.f1_scores = self.f1_score_class()
        
        self.avg_accuracy = np.mean(self.accuracies)
        self.avg_precision = np.mean(self.precisions)
        self.avg_recall = np.mean(self.recalls)
        self.avg_f1_score = np.mean(self.f1_scores)
        
        return

    def accuracy_class(self):
        accuracy = np.zeros(self.num_classes)
        for i in range(self.num_classes):
            correct = self.cm[i, i]  # Previsões corretas para a classe 'i'
            total = np.sum(self.cm[i, :])  # Total de previsões feitas para a classe 'i'
            accuracy[i] = correct / total if total > 0 else 0
        return np.round(accuracy, 4)

    def precision_class(self):
        precision = np.zeros(self.num_classes)
        for i in range(self.num_classes):
            true_positives = self.cm[i, i]
            predicted_positives = np.sum(self.cm[:, i])
            precision[i] = true_positives / predicted_positives if predicted_positives > 0 else 0
        return np.round(precision, 4)

    def recall_class(self):
        recall = np.zeros(self.num_classes)
        for i in range(self.num_classes):
            true_positives = self.cm[i, i]
            actual_positives = np.sum(self.cm[i, :])
            recall[i] = true_positives / actual_positives if actual_positives > 0 else 0
        return np.round(recall, 4)

    def f1_score_class(self):
        precision = self.precision_class()
        recall = self.recall_class()
        f1 = 2 * (precision * recall) / (precision + recall)
        return np.round(f1, 4)


    def __repr__(self):
        cm_prob_str = np.array2string(self.cm_prob, precision=2, suppress_small=True)
        cm_str = np.array2string(self.cm, precision=2, suppress_small=True)
        
        return (f"Annotator(id={self.id}, seed={self.seed}, num_classes={self.num_classes})\n"
                f"Reputations: {self.reputations}\n"
                f"Scores: {self.rating_scores}\n"
                f"Accuracies per Class: {self.accuracies}\n"
                f"Precisions per Class: {self.precisions}\n"
                f"Recalls per Class: {self.recalls}\n"
                f"F1-Scores per Class: {self.f1_scores}\n"
                f"Avg Reputation: {self.avg_reputation:.4f}\n"
                f"Avg Accuracy: {self.avg_accuracy:.4f}\n"
                f"Avg Precision: {self.avg_precision:.4f}\n"
                f"Avg Recall: {self.avg_recall:.4f}\n"
                f"Avg F1-Score: {self.avg_f1_score:.4f}\n"
                f"Current Answer: {self.current_answer}\n"
                f"Confusion Matrix Probabilities (cm_prob):\n{cm_prob_str}\n"
                f"Confusion Matrix of Answers (cm):\n{cm_str}\n")
    
    def repr_cm_prob(self):
        cm_prob_str = np.array2string(self.cm_prob, precision=2, suppress_small=True)
        return (f"Annotator(id={self.id}, seed={self.seed}, num_classes={self.num_classes})\n"
                f"Confusion Matrix Probabilities (cm_prob):\n{cm_prob_str}\n")