import numpy as np
import matplotlib.pyplot as plt
from modAL.models import ActiveLearner
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from modAL.uncertainty import uncertainty_sampling
import torch
import torch.optim as optim
import torch.nn as nn
from entities.LeNet5 import LeNet5
from torch.utils.data import Subset, DataLoader, random_split, ConcatDataset
from torchvision import datasets, transforms
from collections import Counter
from env import *
from torchvision.transforms import functional as F

def get_label_distribution(dataset):
    labels = [dataset[i][1] for i in range(len(dataset))]
    return Counter(labels)

def plot_distribution(distribution, split_name, colors=['royalblue', 'tomato', 'goldenrod']
                      ):
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





def evaluate_model(model, dataloader, device):
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Métricas gerais
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    # Matriz de confusão
    cm = confusion_matrix(all_labels, all_preds)

    # Sensitivity (Recall para cada classe)
    sensitivity_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0)

    # Specificity (calculada a partir da matriz de confusão)
    specificity_per_class = []
    for i in range(len(cm)):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificity_per_class.append(specificity)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted): {recall:.4f}")
    print(f"F1 Score (weighted): {f1:.4f}")
    print("\nSensitivity per class:", sensitivity_per_class)
    print("Specificity per class:", specificity_per_class)
    print("\nConfusion Matrix:\n", cm)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "sensitivity_per_class": sensitivity_per_class,
        "specificity_per_class": specificity_per_class,
        "confusion_matrix": cm
    }



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