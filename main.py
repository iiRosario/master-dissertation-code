import os
import numpy as np
import torch
from modAL.models import ActiveLearner
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader, ConcatDataset
from torchvision import datasets, transforms
from entities.Committee import Committee
from entities.LeNet5 import LeNet5
from env import *
from collections import Counter
from utils.utils import *
import warnings
import argparse

warnings.filterwarnings("ignore", category=FutureWarning)
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

DATASET_IN_USE = "None"
QUERY_STRATEGY_IN_USE = "None"
ORACLE_ANSWER_IN_USE = "None"
EXPERTISE_IN_USE = "None"
ORACLE_SIZE_IN_USE = 0
RATING_FLAG = "None"

def create_results_dir(seed):
    if QUERY_STRATEGY_IN_USE == margin_sampling:
        results_dir_name = f"margin_sampling"
    elif QUERY_STRATEGY_IN_USE == entropy_sampling:
        query_strategy = f"entropy_sampling"
    elif QUERY_STRATEGY_IN_USE == uncertainty_sampling:
        query_strategy = f"uncertainty_sampling"
    else:
        query_strategy = f"random_sampling"
    
    if ORACLE_ANSWER_IN_USE == ORACLE_ANSWER_REPUTATION:
        results_dir_path = os.path.join(RESULTS_PATH, DATASET_IN_USE, query_strategy, ORACLE_ANSWER_IN_USE, RATING_FLAG, str(ORACLE_SIZE_IN_USE), EXPERTISE_IN_USE, f"results_{seed}")

    else:
        results_dir_path = os.path.join(RESULTS_PATH, DATASET_IN_USE, query_strategy, ORACLE_ANSWER_IN_USE, f"results_{seed}")

    os.makedirs(results_dir_path, exist_ok=True)
    return results_dir_path


def select_answer_type(oracle, true_labels):

    if(ORACLE_ANSWER_IN_USE != ORACLE_ANSWER_GROUND_TRUTH):
        answers = []
        for true_label in true_labels:
            if(ORACLE_ANSWER_IN_USE == ORACLE_ANSWER_REPUTATION): 
                ans = oracle.weight_reputation_answer(true_target=true_label)
            elif(ORACLE_ANSWER_IN_USE == ORACLE_ANSWER_RANDOM):
                ans = oracle.random_answer(true_target=true_label)
            answers.append(ans)
        oracle_labels = torch.tensor(answers, dtype=true_labels.dtype)

    elif(ORACLE_ANSWER_IN_USE == ORACLE_ANSWER_GROUND_TRUTH):
        oracle_labels = true_labels
    
    return oracle_labels

def init_active_learning_pool(train_loader, val_loader, test_loader, seed):
    # --- setup de diretórios e CSV ---
    results_path = create_results_dir(seed)
    plots_path = os.path.join(results_path, "plots")
    os.makedirs(plots_path, exist_ok=True)

    results_file = f"results_{seed}.csv"
    csv_path = os.path.join(results_path, results_file)
    if os.path.exists(csv_path):
        os.remove(csv_path)

    # --- extrair dados dos loaders ---
    x_train, y_train = extract_data(train_loader)
    x_val, y_val     = extract_data(val_loader)
    x_test, y_test   = extract_data(test_loader)
    
    # Embaralha o inicial
    torch.manual_seed(seed)
    perm = torch.randperm(len(x_train))
    x_train, y_train = x_train[perm], y_train[perm]

    # Split inicial e pool
    x_init, x_unlabeled, y_init, y_unlabeled = train_test_split(x_train, y_train, 
                                                                train_size=INIT_TRAINING_PERCENTAGE,
                                                                stratify=y_train,
                                                                random_state=seed)

    # Visualiza distribuição inicial
    plot_distribution_2(Counter(y_init.tolist()), "init_train", CLASS_COLORS, plots_path)
    plot_distribution_2(Counter(y_unlabeled.tolist()), "unlabeled_train", CLASS_COLORS, plots_path)
    save_class_distributions_to_csv_2(Counter(y_init.tolist()), Counter(y_unlabeled.tolist()), path=plots_path)

    # --- inicializa modelo e learner ---
    model = LeNet5(device=device, dataset=DATASET_IN_USE).to(device)
    learner = ActiveLearner(
        estimator=model,
        query_strategy=QUERY_STRATEGY_IN_USE,
        X_training=x_init, y_training=y_init
    )

    # Avaliação inicial
    init_metrics = learner.estimator.evaluate(x_val, y_val)
    write_metrics_to_csv(results_path, results_file, cycle=0, oracle_label=-1, ground_truth_label=-1,  metrics=init_metrics, oracle_cm=-1, oracle_iterations=-1)

    oracle = Committee(size=ORACLE_SIZE_IN_USE, seed=seed, expertise=EXPERTISE_IN_USE, results_path=results_path, rating_flag=RATING_FLAG)
    
    # --- loop de Active Learning em batch ---
    for cycle in range(NUM_CYCLES):
        print(f"========================\nCycle {cycle + 1}/{NUM_CYCLES}")

        # Query por batch de instâncias
        query_idx, query_instances = learner.query(x_unlabeled, n_instances=POOL_SIZE)

        # Seleciona imagens e rótulos verdadeiros
        x_query = x_unlabeled[query_idx]
        true_labels = y_unlabeled[query_idx]

        oracle_labels = select_answer_type(oracle, true_labels)

        learner.teach(X=x_query, y=oracle_labels)
        print(f"learner DL size: {len(learner.y_training)}")
        
        # Remove do pool as instâncias já rotuladas
        x_unlabeled = np.delete(x_unlabeled, query_idx, axis=0)
        y_unlabeled = np.delete(y_unlabeled, query_idx, axis=0)

        # Avalia no validation ou test
        if cycle + 1 == NUM_CYCLES:
            metrics = learner.estimator.evaluate(x_test, y_test)
        else:
            metrics = learner.estimator.evaluate(x_val, y_val)

        avg_acc = avg_metric(metrics, 'accuracy_per_class')
        print(f"   AVG Acc: {avg_acc:.4f}\n" +
              f"   AVG Precision: {avg_metric(metrics, 'precision_per_class'):.4f}")

        # Salva métricas
        list_oracle_labels = [oracle_labels[i].item() for i in range(len(query_idx))]
        list_true_labels = [true_labels[i].item() for i in range(len(query_idx))]

        write_metrics_to_csv(results_path, results_file, cycle=cycle + 1,
                            oracle_label=list_oracle_labels,
                            ground_truth_label=list_true_labels,
                            metrics=metrics,
                            oracle_cm=oracle.cm,
                            oracle_iterations=oracle.labeling_iteration)
    
    # --- pós-processamento ---
    plot_all_metrics_over_cycles(csv_path, plots_path, seed)
    print("Done for seed:", seed)



def init_perm_statistic(train_loader, val_loader, test_loader):
    
    for seed in range(30):
        print(f"\n\n\n========== AL =============")
        print(f"Dataset: {DATASET_IN_USE}")
        print(f"Query Strategy: {QUERY_STRATEGY_IN_USE}")
        print(f"Oracle Size: {ORACLE_SIZE_IN_USE}")
        print(f"Oracle Answer: {ORACLE_ANSWER_IN_USE}")
        print(f"Annotator Expertise: {EXPERTISE_IN_USE}")
        print(f"Rating Flag: {RATING_FLAG}")
        print(f"Learning Rate: {LEARNING_RATE}")
        print(f"Epochs: {EPHOCS}")
        print(f"Init training (%): {INIT_TRAINING_PERCENTAGE * 100}%")
        print(f"Running on seed: {seed}")
        
        # Inicializar o modelo e o ciclo de aprendizado ativo
        init_active_learning_pool(train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, seed=seed)

def init_perm_oracle_answer(train_loader, val_loader, test_loader):
    global ORACLE_ANSWER_IN_USE
    global EXPERTISE_IN_USE
    global ORACLE_SIZE_IN_USE
    global RATING_FLAG 

    # random_answer , reputation_based. ground_truth
    for ORACLE_ANSWER_IN_USE in ORACLE_ANSWERS:
        if ORACLE_ANSWER_IN_USE  == ORACLE_ANSWER_REPUTATION:
            
            for ORACLE_SIZE_IN_USE in ORACLE_SIZES: # 5, 15, 30
                 
                for EXPERTISE_IN_USE in EXPERTISES: # L, M, H, R
                    
                    for RATING_FLAG in RATINGS_PERMUTATIONS: # with_rating, without_rating
                        init_perm_statistic(train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)
        else:
            init_perm_statistic(train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)
 
            

def init_perm_query_strategy(train_loader, val_loader, test_loader):
    global QUERY_STRATEGY_IN_USE
    #for QUERY_STRATEGY_IN_USE in QUERY_STRATEGIES:
    #    init_perm_oracle_answer(train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)
   
    QUERY_STRATEGY_IN_USE = UNCERTAINTY_SAMPLING
    init_perm_oracle_answer(train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)


def main(dataset):

    global DATASET_IN_USE
    DATASET_IN_USE = dataset

    path_dir = os.path.join(RESULTS_PATH, DATASET_IN_USE)

    transform_28_28 = transform = transforms.Compose([transforms.Pad(2), 
                                                    transforms.ToTensor(), 
                                                    transforms.Normalize((0.5,), (0.5,))])
    transform_32_32 = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
   
    if DATASET_IN_USE == DATASET_MNIST:
        train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform_28_28)
        test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform_28_28)

    elif DATASET_IN_USE == DATASET_MNIST_FASHION:
        train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_28_28)
        test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_28_28)

    elif DATASET_IN_USE == DATASET_CIFAR_10:
        train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_32_32)
        test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_32_32)

    elif DATASET_IN_USE == DATASET_EMNIST_DIGITS:
        train_data = datasets.EMNIST(root='./data', split='digits', train=True, download=True, transform=transform_28_28)
        test_data = datasets.EMNIST(root='./data', split='digits', train=False, download=True, transform=transform_28_28)
    
    elif DATASET_IN_USE == DATASET_EMNIST_LETTERS:
        train_data = datasets.EMNIST(root='./data', split='letters', train=True, download=True, transform=transform_28_28)
        test_data = datasets.EMNIST(root='./data', split='letters', train=False, download=True, transform=transform_28_28)
    
    elif DATASET_IN_USE == DATASET_SVHN:
        train_data = datasets.SVHN(root='./data', split='train', download=True, transform=transform)
        test_data = datasets.SVHN(root='./data', split='test', download=True, transform=transform)
    else:
        raise ValueError(f"Invalid Dataset : {DATASET_IN_USE}")
    
    plot_sample_images(dataset=test_data, classes=CLASSES, num_samples=5, save_path=path_dir)    

    train_data = filter_classes(train_data, classes=CLASSES)
    test_data = filter_classes(test_data, classes=CLASSES)

    full_dataset = ConcatDataset([train_data, test_data])

    total_size = len(full_dataset)
    train_size = int(TRAIN_SIZE_PERCENTAGE * total_size)
    test_size = int(TEST_SIZE_PERCENTAGE * total_size)
    val_size = total_size - train_size - test_size

    # Índices sequenciais
    all_indices = list(range(total_size))
    train_indices = all_indices[:train_size]
    test_indices = all_indices[train_size:train_size + test_size]
    val_indices = all_indices[train_size + test_size:]

    # Subsets finais
    train_set = Subset(full_dataset, train_indices)
    val_set = Subset(full_dataset, val_indices)
    test_set = Subset(full_dataset, test_indices)

    train_dist = get_label_distribution(train_set)
    val_dist = get_label_distribution(val_set)
    test_dist = get_label_distribution(test_set)
    
    save_class_distributions_to_csv(train_dist, val_dist, test_dist, path_dir)
    plot_distribution(dataset=DATASET_IN_USE, distribution=train_dist, split_name="Train", save_path=path_dir, colors=CLASS_COLORS)
    plot_distribution(dataset=DATASET_IN_USE, distribution=val_dist, split_name="Validation", save_path=path_dir, colors=CLASS_COLORS)
    plot_distribution(dataset=DATASET_IN_USE, distribution=test_dist, split_name="Test", save_path=path_dir, colors=CLASS_COLORS)

    
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=True)

    #init
    init_perm_query_strategy(train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)

    
def test_oracle(dataset):
    global DATASET_IN_USE
    global QUERY_STRATEGY_IN_USE
    global ORACLE_ANSWER_IN_USE
    global EXPERTISE_IN_USE
    global ORACLE_SIZE_IN_USE
    
    DATASET_IN_USE = dataset
    QUERY_STRATEGY_IN_USE = UNCERTAINTY_SAMPLING
    ORACLE_ANSWER_IN_USE = ORACLE_ANSWER_REPUTATION
    EXPERTISE_IN_USE = EXPERT_EXPERTISE
    ORACLE_SIZE_IN_USE = ORACLE_SIZE_LARGE


    results_path = create_results_dir(seed=1)
    committee = Committee(size=ORACLE_SIZE_IN_USE, seed=30, expertise=EXPERTISE_IN_USE, results_path = results_path)
    
    for i in range(1):
        print("=======================================================")
        true_label = i % len(CLASSES)
        ans = committee.weight_reputation_answer(true_target=true_label)
        

    #print(committee.repr_cm())
    #committee.compute_and_print_metrics()

    """ for ann in committee.annotators:
        print(ann.repr_cm_prob()) """

if __name__ == "__main__":
    #test_oracle()
    parser = argparse.ArgumentParser(description="Script que aceita um dataset como argumento.")
    parser.add_argument("--dataset", type=str, required=True, help="Nome do dataset a ser usado")
    args = parser.parse_args()
    
    #test_oracle(args.dataset)
    main(args.dataset)
    


    