import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from modAL.models import ActiveLearner
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader, random_split, ConcatDataset
from torchvision import datasets, transforms
from entities.Annotator import Annotator
from entities.Committee import Committee
from entities.LeNet5 import LeNet5
from env import *
from collections import Counter
from utils.DataManager import *
from utils.utils import *
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"

print("Using device:", device)

DATASET_IN_USE = "None"
QUERY_STRATEGY_IN_USE = "None"
ORACLE_ANSWER_IN_USE = "None"

def create_results_dir():
    if QUERY_STRATEGY_IN_USE == margin_sampling:
        results_dir_name = f"margin_sampling"
    elif QUERY_STRATEGY_IN_USE == entropy_sampling:
        results_dir_name = f"entropy_sampling"
    elif QUERY_STRATEGY_IN_USE == uncertainty_sampling:
        results_dir_name = f"uncertainty_sampling"
    else:
        results_dir_name = f"random_sampling"
    
    results_dir_path = os.path.join(RESULTS_PATH, DATASET_IN_USE, ORACLE_ANSWER_IN_USE, results_dir_name)
    os.makedirs(results_dir_path, exist_ok=True)
    
    return results_dir_path




def init_active_learning_pool(train_loader, val_loader, test_loader, seed):
    #CREATE RESULTS DIR
    results_path = create_results_dir()
    plots_path = os.path.join(results_path, "plots")
    os.makedirs(plots_path, exist_ok=True)
    results_file_name = f"results_{seed}.csv"
    results_csv_path = os.path.join(results_path, results_file_name)
    if os.path.exists(results_csv_path): os.remove(results_csv_path)

    # Obter x_train, y_train, x_val, y_val, x_test, y_test
    x_train, y_train = extract_data(train_loader)
    x_val, y_val = extract_data(val_loader)
    x_test, y_test = extract_data(test_loader)

    torch.manual_seed(seed)  
    indices = torch.randperm(len(x_train))  # Shuffle reprodutível
    x_train = x_train[indices]
    y_train = y_train[indices]
    
    x_init_train, x_rest_train, y_init_train, y_rest_train = train_test_split(x_train, y_train, train_size=INIT_TRAINING_PERCENTAGE, stratify=y_train, random_state=seed)

    
    plot_distribution_2(Counter(y_init_train.tolist()), "init_train", CLASS_COLORS, plots_path)
    plot_distribution_2(Counter(y_rest_train.tolist()), "rest_train", CLASS_COLORS, plots_path)

    model = LeNet5(device=device, dataset=DATASET_IN_USE).to(device)
    learner = ActiveLearner(
        estimator = model,
        query_strategy=QUERY_STRATEGY_IN_USE,
        X_training=x_init_train, y_training=y_init_train
    )
    
    init_results = learner.estimator.evaluate(x_val, y_val)    
    write_metrics_to_csv(csv_path=results_path, csv_name=results_file_name, cycle=0, oracle_label=-1, ground_truth_label=-1, metrics=init_results)
    
    oracle = Committee(annotators=[], seed=seed)

    for cycle in range(NUM_CYCLES):
        print("==========================")
        print(f"Cycle {cycle + 1}/{NUM_CYCLES}")


        x_query = []
        y_query = []
        oracle_labels = []
        true_labels = []
        for i in range(POOL_SIZE):
            query_idx, query_instance = learner.query(x_rest_train)
            query_image = x_rest_train[query_idx]
            true_label = y_rest_train[query_idx]

            if ORACLE_ANSWER_IN_USE == ORACLE_ANSWER_REPUTATION:
                continue
            elif ORACLE_ANSWER_IN_USE == ORACLE_ANSWER_GROUND_TRUTH:
                oracle_label = true_label.item()
                target = oracle_label  # já como int
            else:
                oracle_label = oracle.random_answer()
                target = oracle_label

            # Se ainda for tensor, converter para numpy
            if isinstance(query_image, torch.Tensor):
                query_image = query_image.numpy()

            x_query.append(query_image)
            y_query.append(target)

            x_rest_train = np.delete(x_rest_train, query_idx, axis=0)
            y_rest_train = np.delete(y_rest_train, query_idx, axis=0)

            oracle_labels.append(oracle_label)
            true_labels.append(true_label.item())

        # Depois do loop:
        x_query = np.array(x_query)
        y_query = np.array(y_query)

        learner.teach(X=x_query, y=y_query)
        print("learner y size: " + len(learner.y_training))
    
        if(cycle + 1 == NUM_CYCLES):
            print("FINAL CYCLE")
            metrics = learner.estimator.evaluate(x_test, y_test)    
        else:
            metrics = learner.estimator.evaluate(x_val, y_val)    

        print(f"AVG Accuracy: {avg_metric(metrics, 'accuracy_per_class'):.4f} | AVG Precision: {avg_metric(metrics, 'precision_per_class'):.4f}")

        write_metrics_to_csv(csv_path=results_path, csv_name=results_file_name, cycle=cycle+1, oracle_label=oracle_labels, ground_truth_label=true_labels, metrics=metrics)
    
    
    csv_path = os.path.join(results_path, results_file_name)
    plot_all_metrics_over_cycles(csv_path=csv_path, plot_path=plots_path, seed=seed)
    

    print("DONE! for seed: ", seed) 



def init_active_learning(train_loader, val_loader, test_loader, seed):
    #CREATE RESULTS DIR
    results_path = create_results_dir()
    plots_path = os.path.join(results_path, "plots")
    os.makedirs(plots_path, exist_ok=True)
    results_file_name = f"results_{seed}.csv"
    results_csv_path = os.path.join(results_path, results_file_name)
    if os.path.exists(results_csv_path): os.remove(results_csv_path)

    # Obter x_train, y_train, x_val, y_val, x_test, y_test
    x_train, y_train = extract_data(train_loader)
    x_val, y_val = extract_data(val_loader)
    x_test, y_test = extract_data(test_loader)

    torch.manual_seed(seed)  
    indices = torch.randperm(len(x_train))  # Shuffle reprodutível
    x_train = x_train[indices]
    y_train = y_train[indices]
    
    x_init_train, x_rest_train, y_init_train, y_rest_train = train_test_split(x_train, y_train, train_size=INIT_TRAINING_PERCENTAGE, stratify=y_train, random_state=seed)

    
    plot_distribution_2(Counter(y_init_train.tolist()), "init_train", CLASS_COLORS, plots_path)
    plot_distribution_2(Counter(y_rest_train.tolist()), "rest_train", CLASS_COLORS, plots_path)

    model = LeNet5(device=device, dataset=DATASET_IN_USE).to(device)
    learner = ActiveLearner(
        estimator = model,
        query_strategy=QUERY_STRATEGY_IN_USE,
        X_training=x_init_train, y_training=y_init_train
    )
    
    init_results = learner.estimator.evaluate(x_val, y_val)    
    write_metrics_to_csv(csv_path=results_path, csv_name=results_file_name, cycle=0, oracle_label=-1, ground_truth_label=-1, metrics=init_results)
    
    oracle = Committee(annotators=[], seed=seed)

    for cycle in range(NUM_CYCLES):
        print("==========================")
        print(f"Cycle {cycle + 1}/{NUM_CYCLES}")

       
        query_idx, query_instance = learner.query(x_rest_train)
        query_image = x_rest_train[query_idx]
        true_label = y_rest_train[query_idx]
       
        if(ORACLE_ANSWER_IN_USE == ORACLE_ANSWER_REPUTATION):
            continue
        elif(ORACLE_ANSWER_IN_USE == ORACLE_ANSWER_GROUND_TRUTH):
            oracle_label = true_label.item()
            target = torch.tensor([oracle_label])
        else:
            oracle_label = oracle.random_answer()
            target = torch.tensor([oracle_label])

        learner.teach(X=query_image, y=target)
        
        x_rest_train = np.delete(x_rest_train, query_idx, axis=0)
        y_rest_train = np.delete(y_rest_train, query_idx, axis=0)

        if(cycle + 1 == NUM_CYCLES):
            print("FINAL CYCLE")
            metrics = learner.estimator.evaluate(x_test, y_test)    
        else:
            metrics = learner.estimator.evaluate(x_val, y_val)    

        print(f"AVG Accuracy: {avg_metric(metrics, 'accuracy_per_class'):.4f} | AVG Precision: {avg_metric(metrics, 'precision_per_class'):.4f}")

        write_metrics_to_csv(csv_path=results_path, csv_name=results_file_name, cycle=cycle+1, oracle_label=oracle_label, ground_truth_label=true_label.item(), metrics=metrics)
    
    
    csv_path = os.path.join(results_path, results_file_name)
    plot_all_metrics_over_cycles(csv_path=csv_path, plot_path=plots_path, seed=seed)
    

    print("DONE! for seed: ", seed) 




def init_perm_statistic(train_loader, val_loader, test_loader):
    for seed in range(30):
        print(f"\n\n\n========== AL =============")
        print(f"Dataset: {DATASET_IN_USE}")
        print(f"Query Strategy: {QUERY_STRATEGY_IN_USE}")
        print(f"Oracle Answer: {ORACLE_ANSWER_IN_USE}")
        print(f"Learning Rate: {LEARNING_RATE}")
        print(f"Init Epochs: {INIT_EPHOCS}")
        print(f"Epochs: {EPHOCS}")
        print(f"Init training (%): {INIT_TRAINING_PERCENTAGE * 100}%")
        print(f"Running on seed: {seed}")
        
        # Inicializar o modelo e o ciclo de aprendizado ativo
        init_active_learning(train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, seed=seed)

def init_perm_oracle_answer(train_loader, val_loader, test_loader):
    global ORACLE_ANSWER_IN_USE
    #for ORACLE_ANSWER_IN_USE in ORACLE_ANSWERS:
    #    init_perm_statistic(train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)

    ORACLE_ANSWER_IN_USE = ORACLE_ANSWER_GROUND_TRUTH
    init_perm_statistic(train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)

def init_perm_query_strategy(train_loader, val_loader, test_loader):
    global QUERY_STRATEGY_IN_USE
    for QUERY_STRATEGY_IN_USE in QUERY_STRATEGIES:
        init_perm_oracle_answer(train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)
   
    #QUERY_STRATEGY_IN_USE = ENTROPY_SAMPLING
    #init_perm_oracle_answer(train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)


def main():

    global DATASET_IN_USE
    DATASET_IN_USE = DATASET_CIFAR_10
    path_dir = os.path.join(RESULTS_PATH, DATASET_IN_USE)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
   
    if(DATASET_IN_USE == DATASET_MNIST):
        train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    elif(DATASET_IN_USE == DATASET_MNIST_FASHION):
        train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    elif(DATASET_IN_USE == DATASET_CIFAR_10):
        train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    full_dataset = ConcatDataset([train_data, test_data])
    train_data = filter_classes(train_data, classes=CLASSES)
    test_data = filter_classes(test_data, classes=CLASSES)

    
    #plot_sample_images(full_dataset, classes=CLASSES, num_samples=len(CLASSES))
    
    #noisy_dataset = add_noise_to_data(full_dataset, noise_factor=DATA_AUG_NOISE_FACTOR)
    #plot_sample_images(noisy_dataset, classes=CLASSES, num_samples=6)
    
    #rotated_noisy_dataset = add_bidirectional_rotation(noisy_dataset, angle=25)
    #plot_sample_images(rotated_noisy_dataset, classes=CLASSES, num_samples=6)

    #full_dataset = ConcatDataset([full_dataset, noisy_dataset, rotated_noisy_dataset])  # Junta o dataset original com os dados "noisy"

    total_size = len(full_dataset)
    train_size = int(TRAIN_SIZE_PERCENTAGE * total_size)
    test_size = int(TEST_SIZE_PERCENTAGE * total_size)
    val_size = total_size - train_size - test_size

    train_set, test_set, val_set = random_split(full_dataset, [train_size, test_size, val_size])

    train_dist = get_label_distribution(train_set)
    val_dist = get_label_distribution(val_set)
    test_dist = get_label_distribution(test_set)
    
    print(f"Train, Val, Tes distribution: {get_label_distribution(train_set)}, {val_dist}, {test_dist}")
    plot_distribution(train_dist, "Train", save_path=path_dir, colors=CLASS_COLORS)
    plot_distribution(val_dist, "Validation", save_path=path_dir, colors=CLASS_COLORS)
    plot_distribution(test_dist, "Test", save_path=path_dir, colors=CLASS_COLORS)

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=True)


    init_perm_query_strategy(train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)

    
    

if __name__ == "__main__":
    main() 

    