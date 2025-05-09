from env import *
import numpy as np
from torch.utils.data import Subset
import os
import matplotlib.pyplot as plt
import torch
from collections import Counter
import csv
import pandas as pd
from torchvision.transforms import functional as F
from matplotlib.ticker import MaxNLocator
import matplotlib.cm as cm
from PIL import Image


CLASS_COLORS = ['royalblue', 'tomato', 'goldenrod','mediumseagreen', 'orchid', 'slateblue',  'darkorange', 'turquoise', 'firebrick', 'deeppink']


def get_label_distribution(dataset):
    labels = [dataset[i][1] for i in range(len(dataset))]
    return Counter(labels)

def plot_distribution(dataset, distribution, split_name, save_path='.', colors=CLASS_COLORS):
    classes = sorted(distribution.keys())
    counts = [distribution[c] for c in classes]

    if dataset == "CIFAR-10":
        dataset = "CIFAR 10"
    elif dataset == "MNIST_FASHION":
        dataset = "MNIST Fashion"
    elif dataset == "EMNIST_DIGITS":
        dataset = "EMNIST Digits"


    plt.figure(figsize=(6, 4))
    plt.bar([str(c) for c in classes], counts, color=colors[:len(classes)])
    plt.xlabel('Class')
    plt.ylabel('Nº of Samples')
    plt.title(f'{dataset} dataset distribution - {split_name}')
    plt.tight_layout()

    os.makedirs(save_path, exist_ok=True)

    filename = f"class_distribution_{split_name.lower()}.png"
    full_path = os.path.join(save_path, filename)

    plt.savefig(full_path)
    plt.close()
    print(f"Saved: {full_path}")

def plot_distribution_2(distribution, split_name, colors=CLASS_COLORS, save_path='.'):
    classes = sorted(distribution.keys())
    counts = [distribution[c] for c in classes]

    if colors is None or len(colors) < len(classes):
        # Gera cores únicas usando uma colormap
        cmap = cm.get_cmap('tab10' if len(classes) <= 10 else 'tab20', len(classes))
        colors = [cmap(i) for i in range(len(classes))]

    plt.figure(figsize=(6, 4))
    plt.bar([str(c) for c in classes], counts, color=colors[:len(classes)])
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.title(f'Class Distribution - {split_name}')
    plt.tight_layout()

    filename = f"class_distribution_{split_name.lower()}.png"
    full_path = os.path.join(save_path, filename)
    plt.savefig(full_path)
    plt.close()
    print(f"Saved: {full_path}")

def plot_sample_images(dataset, classes, num_samples=6):
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 15))

    for i in range(num_samples):
        # Pega uma amostra aleatória do dataset
        img, label = dataset[i]
        
        # Se a imagem for 3D (RGB), exibe diretamente
        if img.ndimension() == 3:  # (C, H, W) para CIFAR
            img = img.permute(1, 2, 0)  # Transforma para (H, W, C)

        axes[i].imshow(img)  # Exibe a imagem
        axes[i].axis('off')  # Desliga os eixos
        axes[i].set_title(f"Class: {classes[label.item()]}")
    
    plt.tight_layout()
    plt.show()



def plot_sample_images(dataset, classes, num_samples=5, num_classes=10, save_path='sample_plot.png'):
    """
    Plota e salva uma grade de imagens com uma coluna por classe e uma linha por amostra.

    Parâmetros:
        dataset (torch.utils.data.Dataset): Dataset PyTorch (ex: CIFAR-10, FashionMNIST, MNIST, EMNIST).
        classes (list): Lista de índices das classes a visualizar (1 por coluna).
        num_samples (int): Número de imagens por classe (1 por linha).
        num_classes (int): Número de classes a incluir no plot (limita o tamanho de `classes`).
        save_path (str): Caminho para salvar a figura gerada.
    """

    # Reduzir a lista de classes conforme o limite desejado
    classes = classes[:num_classes]

    # Identificar o nome do dataset
    dataset_name = dataset.__class__.__name__

    if 'CIFAR10' in dataset_name:
        all_class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
    elif 'FashionMNIST' in dataset_name:
        all_class_names = [
            'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
        ]
    elif 'MNIST' in dataset_name and 'Fashion' not in dataset_name:
        all_class_names = [str(i) for i in range(10)]
    elif 'EMNIST' in dataset_name:
        all_class_names = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
    elif 'SVHN' in dataset_name:
        all_class_names = [str(i) for i in range(10)]
    else:
        all_class_names = [str(i) for i in range(100)]

    is_emnist_letters = 'EMNIST' in dataset_name and max(classes) > 9

    class_names = {}
    for i in classes:
        if is_emnist_letters:
            class_names[i] = all_class_names[i - 1]
        else:
            class_names[i] = all_class_names[i]

    class_images = {cls: [] for cls in classes}
    
    for img, label in dataset:
        label = label.item() if isinstance(label, torch.Tensor) else label
        if label in classes and len(class_images[label]) < num_samples:
            class_images[label].append(img)
        if all(len(class_images[cls]) >= num_samples for cls in classes):
            break

    fig, axes = plt.subplots(num_samples, len(classes), figsize=(len(classes)*2.5, num_samples*2.5))

    if num_samples == 1:
        axes = axes.reshape(1, -1)
    if len(classes) == 1:
        axes = axes.reshape(-1, 1)

    for col, cls in enumerate(classes):
        for row, img in enumerate(class_images[cls]):
            ax = axes[row, col]

            if isinstance(img, Image.Image):
                img = to_tensor(img)

            if img.min() < 0:
                img = img * 0.5 + 0.5

            if img.ndimension() == 3:
                img = img.permute(1, 2, 0)

            ax.imshow(img.squeeze(), cmap='gray' if img.ndimension() == 2 or img.shape[-1] == 1 else None)
            ax.axis('off')  # <-- Aqui desativa os eixos

        axes[0, col].set_title(class_names[cls], fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    plt.close()

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




    


def write_metrics_to_csv(csv_path, csv_name, cycle, oracle_label, ground_truth_label, metrics, oracle_cm, oracle_iterations):
    fieldnames = ["cycle", "accuracy_per_class", "precision_per_class", 
                  "recall_per_class", "f1_score_per_class", "sensitivity_per_class",
                  "specificity_per_class", "confusion_matrix", 
                  "oracle_label", "ground_truth_label", 
                  "oracle_confusion_matrix", "oracle_iterations"]

    
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
            "ground_truth_label": ground_truth_label,
            "oracle_confusion_matrix": oracle_cm,
            "oracle_iterations": oracle_iterations
        })

    

# ============================= PLOTAR MÉTRICAS ============================
def plot_metric_over_cycles(csv_path, plot_path, variable, filename):
    df = pd.read_csv(csv_path)

    # Converter strings da coluna para listas de float
    def parse_list(row):
        return [float(x) for x in row.strip("[]").replace("'", "").split(',')]

    df[variable] = df[variable].apply(parse_list)

    # Calcular média da métrica por ciclo
    df['mean'] = df[variable].apply(np.mean)

    cycles = df['cycle']
    means = df['mean']

    # Plotar
    plt.figure(figsize=(10, 6))
    plt.plot(cycles, means, label=f'Average {variable}', color='blue')

    # Forçar eixo X com valores inteiros
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Ajustar eixos e título
    plt.title(f'{variable.replace("_", " ").title()} over AL Iterations')
    plt.xlabel('Iteration')
    plt.ylabel(variable.replace("_", " ").title())
    plt.legend()
    plt.grid(True)

    # Garantir que o diretório de destino existe
    os.makedirs(plot_path, exist_ok=True)

    # Salvar figura
    plot_file = os.path.join(plot_path, f'{filename}.png')
    plt.savefig(plot_file)
    plt.close()

    print(f"Plot saved to: {plot_file}")



def plot_all_metrics_over_cycles(csv_path, plot_path, seed):

    accuracy_plot_path = os.path.join(plot_path, "avg_accuracy")     
    os.makedirs(accuracy_plot_path, exist_ok=True)
    
    precision_plot_path = os.path.join(plot_path, "avg_precision")     
    os.makedirs(accuracy_plot_path, exist_ok=True)
    
    
    plot_metric_over_cycles(csv_path=csv_path, 
                            plot_path=accuracy_plot_path, 
                            variable="accuracy_per_class",
                            filename=f"accuracy_{seed}")
    
    plot_metric_over_cycles(csv_path=csv_path, 
                            plot_path=precision_plot_path, 
                            variable="precision_per_class",
                            filename=f"precision_{seed}")




def avg_metric(metrics, variable):
    variable_list = [float(a) for a in metrics[variable]]
    avg_variable = sum(variable_list) / len(variable_list)
    return avg_variable

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



def save_class_distributions_to_csv(train_dist, val_dist, test_dist, path):
    # Garante que o diretório existe
    filename='class_distribution.csv'
    path = os.path.join(path, filename) 
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
      # Remove o ficheiro se já existir
    if os.path.exists(path):
        os.remove(path)

    classes = sorted(set(train_dist.keys()) | set(val_dist.keys()) | set(test_dist.keys()))

    with open(path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Class', 'Train', 'Validation', 'Test'])  # Cabeçalho

        for cls in classes:
            train_count = train_dist.get(cls, 0)
            val_count = val_dist.get(cls, 0)
            test_count = test_dist.get(cls, 0)
            writer.writerow([cls, train_count, val_count, test_count])



def save_class_distributions_to_csv_2(init_train_dist, rest_train_dist, path):
    # Garante que o diretório existe
    filename='train_class_distribution.csv'
    path = os.path.join(path, filename) 
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
      # Remove o ficheiro se já existir
    if os.path.exists(path):
        os.remove(path)

    classes = sorted(set(init_train_dist.keys()) | set(rest_train_dist.keys()))

    with open(path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Class', 'init_train_dist', 'rest_train_dist'])  # Cabeçalho

        for cls in classes:
            train_count = init_train_dist.get(cls, 0)
            val_count = rest_train_dist.get(cls, 0)
            writer.writerow([cls, train_count, val_count])


