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

import shutil

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def print_distribuition(x, y):

    y = y.numpy() if isinstance(y, torch.Tensor) else y
    print(f"x_init_train: {len(x)}, y_init_train:", Counter(y))
   
def create_results_dir(seed):
    if QUERY_STRATEGY == margin_sampling:
        results_dir_name = f"results_margin_sampling_{seed}"
    elif QUERY_STRATEGY == entropy_sampling:
        results_dir_name = f"results_entropy_sampling_{seed}"
    else:
        results_dir_name = f"results_uncertainty_sampling_{seed}"
    
    results_dir_path = os.path.join(RESULTS_PATH, ORACLE_ANSWER, results_dir_name)

    # Remove a pasta se ela já existir
    if os.path.exists(results_dir_path):
        shutil.rmtree(results_dir_path)

    # Cria a nova pasta
    os.makedirs(results_dir_path, exist_ok=True)
    
    return results_dir_path



def init_model_lenet5(device=device, epochs=INIT_TRAINING_EPHOCHS, lr=INIT_LEARNING_RATE, batch_size=64):
    model = LeNet5(device)
    output_dir = MODELS
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "lenet_base.pth")
    # Save the untrained model weights
    torch.save(model.state_dict(), save_path)
    print(f"LeNet-5 base model saved to: {save_path}")
    return model



def initial_training(model, train_loader, val_loader, train_percentage=0.1, num_epochs=2, lr=0.9, seed=0):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    total_train_size = len(train_loader.dataset)
    train_size = int(train_percentage * total_train_size)

    # Criar subset com base no train_percentage
    np.random.seed(seed)
    indices = np.random.choice(total_train_size, train_size, replace=False)

    # Mostrar distribuição das classes no subset
    label_dist = Counter([train_loader.dataset[i][1] for i in indices])
    print("Initial training class distribution:", label_dist)

    train_subset = Subset(train_loader.dataset, indices)
    train_subset_loader = DataLoader(train_subset, batch_size=train_loader.batch_size, shuffle=True)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_subset_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_subset_loader)
        train_losses.append(avg_train_loss)

        # Avaliação no conjunto de validação
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    return model, train_losses, val_losses

# Active learning loop
def init_active_learning(train_loader, val_loader, test_loader, seed):
    
    
    # Obter x_train, y_train, x_val, y_val, x_test, y_test
    x_train, y_train = extract_data(train_loader)
    x_val, y_val = extract_data(val_loader)
    x_test, y_test = extract_data(test_loader)

    #VERIFICAR SE O DATASET DE TRAIN EStá SHUFFLED
    torch.manual_seed(seed)  
    indices = torch.randperm(len(x_train))  # Shuffle reprodutível
    x_train = x_train[indices]
    y_train = y_train[indices]

    x_init_train, x_rest_train, y_init_train, y_rest_train = train_test_split(x_train, y_train, train_size=INIT_TRAINING_PERCENTAGE, stratify=y_train, random_state=seed)

    x_train_labeled = []
    y_train_labeled = []


    torch.manual_seed(seed)  
    indices = torch.randperm(len(x_init_train))  # Shuffle reprodutível
    x_init_train = x_init_train[indices]
    y_init_train = y_init_train[indices]

    model = init_model_lenet5(device=device, epochs=INIT_TRAINING_EPHOCHS, lr=INIT_LEARNING_RATE, batch_size=64)
    learner = ActiveLearner(
        estimator = model,
        query_strategy=QUERY_STRATEGY,  # Using uncertainty sampling as the query strategy
        X_training=x_init_train, y_training=y_init_train
    )
    #CREATE RESULTS DIR
    results_path = create_results_dir(seed) 
    results_file_name = f"results_{seed}.csv"
    init_results = learner.estimator.evaluate(x_val, y_val)    
    write_metrics_to_csv(csv_path=results_path, csv_name=results_file_name, cycle=0, 
                         oracle_label=-1, ground_truth_label=-1, metrics=init_results)
    

    oracle = Committee(annotators=[], seed=seed)

    for cycle in range(NUM_CYCLES):
        print("==========================")
        print(f"Cycle {cycle + 1}/{NUM_CYCLES}")

        # Query da amostra mais incerta 
        query_idx, query_instance = learner.query(x_rest_train)
        
        # Obter imagem e ground truth
        query_image = x_rest_train[query_idx]
        true_label = y_rest_train[query_idx]
        
        if(ORACLE_ANSWER == "reputation"):
            continue
        elif(ORACLE_ANSWER == "ground_truth"):
            oracle_label = true_label.item()
            target = torch.tensor([oracle_label])
        else:
            oracle_label = oracle.random_answer()
            target = torch.tensor([oracle_label])

        learner.teach(X=query_image, y=target)

        # Remover amostra consultada do conjunto de treino e adicionar aos conjuntos rotulados
        x_rest_train = np.delete(x_rest_train, query_idx, axis=0)
        y_rest_train = np.delete(y_rest_train, query_idx, axis=0)

        # Adicionar a amostra rotulada ao conjunto de treino rotulado
        x_train_labeled.append(query_image)
        y_train_labeled.append(true_label)

        if(cycle + 1 == NUM_CYCLES):
            metrics = learner.estimator.evaluate(x_test, y_test)    
        else:
            metrics = learner.estimator.evaluate(x_val, y_val)    

        write_metrics_to_csv(csv_path=results_path, csv_name=results_file_name, 
                             cycle=cycle+1, oracle_label=oracle_label, ground_truth_label=true_label.item(), metrics=metrics)
    

    plot_metric_over_cycles(csv_path=os.path.join(results_path, results_file_name), 
                            plot_path=results_path, 
                            variable="accuracy_per_class",
                            filename=f"precision_{seed}")
    print("DONE! for seed: ", seed)





# Main function to set up the environment
def main():

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Carregar e combinar FashionMNIST (train + test)
    train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    
    # Filtrar as classes 0, 1 e 2
    train_data = filter_classes(train_data, classes=CLASSES)
    test_data = filter_classes(test_data, classes=CLASSES)
    
    full_dataset = ConcatDataset([train_data, test_data])
    #plot_sample_images(full_dataset, classes=CLASSES, num_samples=6)
    noisy_dataset = add_noise_to_data(full_dataset, noise_factor=DATA_AUG_NOISE_FACTOR)
    #plot_sample_images(noisy_dataset, classes=CLASSES, num_samples=6)
    rotated_noisy_dataset = add_bidirectional_rotation(noisy_dataset, angle=25)
    #plot_sample_images(rotated_noisy_dataset, classes=CLASSES, num_samples=6)


    full_dataset = ConcatDataset([full_dataset, noisy_dataset, rotated_noisy_dataset])  # Junta o dataset original com os dados "noisy"

    # Cálculo das proporções: 70% treino, 20% teste, 10% validação
    total_size = len(full_dataset)
    train_size = int(TRAIN_SIZE_PERCENTAGE * total_size)
    test_size = int(TEST_SIZE_PERCENTAGE * total_size)
    val_size = total_size - train_size - test_size

    train_set, test_set, val_set = random_split(full_dataset, [train_size, test_size, val_size])

    # Obter distribuições
    train_dist = get_label_distribution(train_set)
    val_dist = get_label_distribution(val_set)
    test_dist = get_label_distribution(test_set)
    
    print(f"Train, Val, Tes distribution: {train_dist}, {val_dist}, {test_dist}")
    plot_distribution(train_dist, "Train", colors=CLASS_COLORS)
    plot_distribution(val_dist, "Validation", colors=CLASS_COLORS)
    plot_distribution(test_dist, "Test", colors=CLASS_COLORS)

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=True)

    for seed in range(30):
        print(f"\n\n\n========== AL =============")
        print(f"Running on seed: {seed}")
        # Inicializar o modelo e o ciclo de aprendizado ativo
        init_active_learning(train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, seed=seed)

if __name__ == "__main__":
    main() 