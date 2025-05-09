from entities.Annotator import Annotator
from env import *
import random
import os
import pandas as pd


class Committee:
    def __init__(self, size=30, seed=-1, expertise=HIGH_EXPERTISE, results_path=None, rating_flag=WITH_RATING):
        self.seed = seed    
        random.seed(seed)
        self.results_path = results_path
        self.labeling_iteration = 0
        self.rating_flag = rating_flag

        self.labeled_samples_class = [0 for _ in range(len(CLASSES))]
        self.cm = np.zeros((len(CLASSES), len(CLASSES)))

        self.size = size

        self.annotators = []
        for i in range(size):
            if self.rating_flag == WITH_RATING:         
                alpha, beta = 0.5, 0.5
            elif self.rating_flag == WITHOUT_RATING:    
                alpha, beta = 0.0, 1.0
            else:                                       
                alpha, beta = 0.5, 0.5  

            #print(f"alpha: {alpha}, beta: {beta}")
            ann = Annotator(id=i, seed=seed+i, num_classes=len(CLASSES), alpha=alpha, beta=beta, expertise=expertise)
            self.annotators.append(ann)
    

    def update_metrics(self, answer, true_target):
        self.cm[true_target, answer] += 1
        

    def random_answer(self, true_target):
        self.labeling_iteration += 1
        committee_answer = random.choice(CLASSES)
        self.update_metrics(answer=committee_answer, true_target=true_target)
        #self.write_annotators()
        return committee_answer
    

    def weight_reputation_answer(self, true_target):
        committee_answer = -1
        self.labeling_iteration += 1

        # GENERATE ANSWERS
        for annotator in self.annotators:
            annotator.answer(true_target)
        
        # RATE OTHERS (skip self)
        if(self.rating_flag == WITH_RATING):
            for who_rates in self.annotators:
                for other in self.annotators:
                    if who_rates is not other:
                        who_rates.rate(other=other, true_label=true_target)
                        #print(f"who_rates_id: {who_rates.id}   other_id: {other.id}")
                        #print(f"who_rates_current_answer: {who_rates.current_answer}   other_current_answer: {other.current_answer}")
            
        sum_reputations = [0.00 for _ in range(len(CLASSES))]
        
        for annotator in self.annotators:
            annotator.update_reputation_per_class(N=self.size, iteration=self.labeling_iteration)
            #print(annotator.rating_scores)          
            class_answer = annotator.current_answer
            class_reputation = annotator.reputations[class_answer]
            sum_reputations[class_answer] += class_reputation

        committee_answer = sum_reputations.index(max(sum_reputations))
        self.update_metrics(answer=committee_answer, true_target=true_target)
        self.write_annotators()
        return committee_answer


    def compute_and_print_metrics(self):
        """
        Computes and prints accuracy, precision, recall, F1 score,
        number of correct predictions (hits) and number of wrong predictions (fails)
        based on the confusion matrix self.cm.
        """
        cm = self.cm
        num_classes = len(CLASSES)

        correct_per_class = np.diag(cm)
        total_true_per_class = cm.sum(axis=1)
        total_pred_per_class = cm.sum(axis=0)

        precision = np.zeros(num_classes)
        recall = np.zeros(num_classes)
        f1_score = np.zeros(num_classes)

        for i in range(num_classes):
            if total_pred_per_class[i] > 0:
                precision[i] = correct_per_class[i] / total_pred_per_class[i]
            else:
                precision[i] = 0.0
            
            if total_true_per_class[i] > 0:
                recall[i] = correct_per_class[i] / total_true_per_class[i]
            else:
                recall[i] = 0.0

            if precision[i] + recall[i] > 0:
                f1_score[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
            else:
                f1_score[i] = 0.0

        avg_precision = np.mean(precision)
        avg_recall = np.mean(recall)
        avg_f1_score = np.mean(f1_score)

        total_correct = correct_per_class.sum()
        total_samples = cm.sum()
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        num_hits = total_correct
        num_fails = total_samples - total_correct

        # Imprimir métricas
        print("\n========== Metrics ==========")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Number of Hits: {int(num_hits)}")
        print(f"Number of Fails: {int(num_fails)}\n")

        print("Per Class Metrics:")
        for i in range(num_classes):
            print(f"Class {i}: Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}, F1-Score: {f1_score[i]:.4f}")
        
        print("\nAverages (Macro):")
        print(f"Average Precision: {avg_precision:.4f}")
        print(f"Average Recall: {avg_recall:.4f}")
        print(f"Average F1-Score: {avg_f1_score:.4f}")
        print("==============================\n")

 
    def write_annotators(self):
        folder = "Annotators"
        output_dir = os.path.join(self.results_path, folder)
        os.makedirs(output_dir, exist_ok=True)

        for annotator in self.annotators:
            filename = f"Annotator_{annotator.id}.csv"
            full_path = os.path.join(output_dir, filename)
            file_exists = os.path.isfile(full_path)

            fieldnames = [
                "iteration", "cm", "cm_ratings", "reputations", "rating_scores", "accuracies"
            ]

            # Formatando as listas
            cm_str = str(annotator.cm)
            cm_ratings_str = str(annotator.cm_ratings)  # Novo campo
            reputations_str = str([round(float(r), 2) for r in annotator.reputations])
            rating_scores_str = str(annotator.rating_scores)
            accuracies_str = str([round(float(a), 4) for a in annotator.accuracies])

            with open(full_path, mode='a', newline='') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

                if not file_exists:
                    writer.writeheader()

                writer.writerow({
                    "iteration": self.labeling_iteration,
                    "cm": cm_str,
                    "cm_ratings": cm_ratings_str,  # Adicionado ao CSV
                    "reputations": reputations_str,
                    "rating_scores": rating_scores_str,
                    "accuracies": accuracies_str
                })

    def __repr__(self):
        return (f"\nCommittee:\n"
                f"Number of Annotators: {len(self.annotators)}\n"
                f"List of Specialists per Class: {self.list_num_specialists}\n")

    def repr_cm(self):
        cm_str = np.array2string(self.cm, precision=2, suppress_small=True)
        return (f"Iteration={self.labeling_iteration}\n" 
                f"Confusion Matrix:\n{cm_str}\n")