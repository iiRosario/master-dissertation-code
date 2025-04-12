import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Subset
from collections import Counter
from env import *
import csv
from torchvision.transforms import functional as F

def get_label_distribution(dataset):
    labels = [dataset[i][1] for i in range(len(dataset))]
    return Counter(labels)

def plot_distribution(distribution, split_name, colors=['royalblue', 'tomato', 'goldenrod']):
    classes = sorted(distribution.keys())
    counts = [distribution[c] for c in classes]

    plt.figure(figsize=(6, 4))
    plt.bar([str(c) for c in classes], counts, color=colors[:len(classes)])
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.title(f'Class Distribution - {split_name}')
    plt.tight_layout()

    filename = f"class_distribution_{split_name.lower()}.png"
    plt.savefig(filename)
    plt.close()
    print(f"Saved: {filename}")

# Função para plotar 3 figuras do dataset para as classes 0, 1 e 2
def plot_sample_images(dataset, classes=[0, 1, 2], num_samples=3):
    # Mapear as classes numéricas para seus respectivos nomes no FashionMNIST
    class_names = ['T-shirt/top', 'Trouser', 'Pullover']  # Nomes das classes 0, 1, 2

    # Criar um dicionário para armazenar imagens de cada classe
    class_images = {cls: [] for cls in classes}
    
    # Encontrar 'num_samples' imagens para cada classe
    for img, label in dataset:
        if label in classes and len(class_images[label]) < num_samples:
            class_images[label].append(img)
        
        # Stop early if we have enough images for all classes
        if all(len(class_images[cls]) >= num_samples for cls in classes):
            break
    
    # Criar as subplots para as imagens por classe (matriz)
    fig, axes = plt.subplots(len(classes), num_samples, figsize=(15, 5))
    
    for i, cls in enumerate(classes):
        for j, img in enumerate(class_images[cls]):
            axes[i, j].imshow(img.squeeze(), cmap='gray')
            axes[i, j].set_xticks([])  # Remover marcações do eixo X
            axes[i, j].set_yticks([])  # Remover marcações do eixo Y
            
        # Definir o título da linha (para todas as imagens dessa classe)
        axes[i, 0].set_ylabel(f"{class_names[cls]}", fontsize=14)
    
    plt.tight_layout()
    plt.show()


# Filter dataset for selected classes (e.g., [0, 1, 2])
def filter_classes(dataset, classes=[0, 1, 2]):
    indices = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        if label in classes:
            indices.append(i)
    return Subset(dataset, indices)


def extract_data(loader):
    data_list = []
    label_list = []
    for data, labels in loader:
        data_list.append(data)       # shape: [batch_size, 1, 28, 28]
        label_list.append(labels)    # shape: [batch_size]
    
    data = torch.cat(data_list, dim=0)     
    labels = torch.cat(label_list, dim=0)
    
    return data, labels



def write_metrics_to_csv(csv_path, csv_name, cycle, oracle_label, ground_truth_label, metrics):
    fieldnames = ["cycle", "accuracy_per_class", "precision_per_class", 
                  "recall_per_class", "f1_score_per_class", "sensitivity_per_class",
                  "specificity_per_class", "confusion_matrix", 
                  "oracle_label", "ground_truth_label"]

    
    # Caminho completo para o ficheiro
    full_csv_path = os.path.join(csv_path, csv_name)
    file_exists = os.path.isfile(full_csv_path)

    with open(full_csv_path, mode='a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        # Escreve o cabeçalho se o ficheiro for novo
        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "cycle": cycle,
            "accuracy_per_class": metrics["accuracy_per_class"],
            "precision_per_class": metrics["precision_per_class"],
            "recall_per_class": metrics["recall_per_class"],
            "f1_score_per_class": metrics["f1_score_per_class"],
            "sensitivity_per_class": metrics["sensitivity_per_class"],
            "specificity_per_class": metrics["specificity_per_class"],
            "confusion_matrix": metrics["confusion_matrix"],
            "oracle_label": oracle_label,
            "ground_truth_label": ground_truth_label
        })

    


############################### DATA AUGMENTATION ###############################
# Função para adicionar ruído aos dados de entrada
def add_noise_to_data(dataset, noise_factor=0.5):
    noisy_data = []
    for img, label in dataset:
        noisy_img = img + noise_factor * torch.randn_like(img)
        noisy_img = torch.clamp(noisy_img, 0., 1.)  # Limitar para [0,1]
        noisy_data.append((noisy_img, label))
    return noisy_data


def add_bidirectional_rotation(dataset, angle=25):
    augmented_data = []

    for img, label in dataset:
        # Rotacionar para a direita (ângulo positivo)
        rotated_right = F.rotate(img, angle)
        augmented_data.append((rotated_right, label))

        # Rotacionar para a esquerda (ângulo negativo)
        rotated_left = F.rotate(img, -angle)
        augmented_data.append((rotated_left, label))

    return augmented_data