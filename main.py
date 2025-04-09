import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from modAL.models import ActiveLearner
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from modAL.uncertainty import uncertainty_sampling
from torch.utils.data import Subset, DataLoader, random_split, ConcatDataset
from torchvision import datasets, transforms
from entities.Annotator import Annotator
from entities.Committee import Committee
from entities.LeNet5 import LeNet5
from utils.Logger import Logger
from env import *
import matplotlib.pyplot as plt
from collections import Counter
from utils.DataManager import *
from utils.utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def init_model_lenet5(device=device):
    model = LeNet5().to(device)
    output_dir = MODELS
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "lenet_base.pth")
    # Save the untrained model weights
    torch.save(model.state_dict(), save_path)
    print(f"LeNet-5 base model saved to: {save_path}")
    return model



def initial_training(model, train_loader, val_loader, train_percentage=0.1, num_epochs=2, lr=0.9):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    total_train_size = len(train_loader.dataset)
    train_size = int(train_percentage * total_train_size)

    # Criar subset com base no train_percentage
    np.random.seed(0)  # Para reprodutibilidade
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
def init_active_learning(train_set, test_set, val_set, train_percentage, seed):
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=True)
    
    model = init_model_lenet5(device=device)
    model, train_losses, val_losses = initial_training(model, 
                                                       train_loader, 
                                                       val_loader, 
                                                       train_percentage=train_percentage, 
                                                       num_epochs=3, 
                                                       lr=0.9)

    learner = ActiveLearner(
        estimator = model,
        query_strategy=uncertainty_sampling  # Using uncertainty sampling as the query strategy
    )

    evaluate_model(learner.estimator, test_loader, device)
    #for cycle in range(NUM_CYCLES):
        #print(f"  Cycle {cycle + 1}/{num_cycles}")

        # Step 1: Query the most uncertain sample using uncertainty sampling
        #query_idx, query_instance = learner.query(train_set.data.numpy())  # Query the most informative sample(s)

        # Step 2: Use the annotator_query() to get the correct label for the queried sample
        #query_image = query_instance[0]  # If querying more than one sample, adjust accordingly
        #query_label = annotator_query(query_image)  # Get the true label from the oracle

        # Step 3: Teach the learner with the labeled sample
        #learner.teach(X=query_instance, y=torch.tensor([query_label]))  # Train the model with the new labeled sample

        # Optionally: For debugging, show the image and its label
        #print(f"   Queried image index: {query_idx}, Label: {query_label}")


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
    plot_sample_images(full_dataset, classes=CLASSES, num_samples=6)

    # Adicionar ruído aos dados para dificultar o aprendizado
    noisy_dataset = add_noise_to_data(full_dataset, noise_factor=DATA_AUG_NOISE_FACTOR)
    plot_sample_images(noisy_dataset, classes=CLASSES, num_samples=6)

    rotated_noisy_dataset = add_bidirectional_rotation(noisy_dataset, angle=25)
    plot_sample_images(rotated_noisy_dataset, classes=CLASSES, num_samples=6)


    full_dataset = ConcatDataset([full_dataset, noisy_dataset, rotated_noisy_dataset])  # Junta o dataset original com os dados "noisy"

    # Cálculo das proporções: 70% treino, 20% teste, 10% validação
    total_size = len(full_dataset)
    train_size = int(TRAIN_SIZE_PERCENTAGE * total_size)
    test_size = int(TEST_SIZE_PERCENTAGE * total_size)
    val_size = total_size - train_size - test_size

    train_set, test_set, val_set = random_split(full_dataset, [train_size, test_size, val_size])

    # DataLoaders
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
    
    # Obter distribuições
    train_dist = get_label_distribution(train_set)
    val_dist = get_label_distribution(val_set)
    test_dist = get_label_distribution(test_set)
    
    
    print(f"Train, Val, Tes distribution: {train_dist}, {val_dist}, {test_dist}")
    # Plotar com múltiplas cores por barra
    
    plot_distribution(train_dist, "Train", colors=CLASS_COLORS)
    plot_distribution(val_dist, "Validation", colors=CLASS_COLORS)
    plot_distribution(test_dist, "Test", colors=CLASS_COLORS)

    init_active_learning(train_set=train_set, test_set=test_set, val_set=val_set,
                         train_percentage=INIT_TRAINING_PERCENTAGE, seed=0)

if __name__ == "__main__":
    main() 