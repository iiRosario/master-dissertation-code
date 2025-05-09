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
import zipfile
import urllib.request
from torchvision.datasets import ImageFolder
import matplotlib.cm as cm
from PIL import Image
from collections import defaultdict
import random

CLASS_COLORS = ['royalblue', 'tomato', 'goldenrod','mediumseagreen', 'orchid', 'slateblue',  'darkorange', 'turquoise', 'firebrick', 'deeppink']

# Caminhos e constantes

def get_label_distribution(dataset):
    labels = [dataset[i][1] for i in range(len(dataset))]
    return Counter(labels)

def plot_distribution(distribution, split_name, save_path='.', colors=CLASS_COLORS, title=""):
    classes = sorted(distribution.keys())
    counts = [distribution[c] for c in classes]

    plt.figure(figsize=(6, 4))
    plt.bar([str(c) for c in classes], counts, color=colors[:len(classes)])
    plt.xlabel('Class')
    plt.ylabel('Nº of Samples')
    plt.title(title)
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


def plot_sample_images(dataset, classes, num_samples=5, num_classes=5, save_path='sample_plot.png'):
    classes = classes[:num_classes]
    class_images = {}

    # Coletar imagens por classe
    for cls in classes:
        class_images[cls] = []

    for img, label in dataset:
        label = label.item() if isinstance(label, torch.Tensor) else label
        if label in class_images and len(class_images[label]) < num_samples:
            class_images[label].append(img)
        if all(len(imgs) >= num_samples for imgs in class_images.values()):
            break

    # Remover classes que não têm imagens suficientes
    class_images = {cls: imgs for cls, imgs in class_images.items() if len(imgs) == num_samples}
    valid_classes = list(class_images.keys())

    # Criar grid de subplots apenas com classes válidas
    fig, axes = plt.subplots(num_samples, len(valid_classes), figsize=(len(valid_classes) * 2.5, num_samples * 2.5))

    # Garantir que axes seja 2D
    if num_samples == 1 and len(valid_classes) == 1:
        axes = np.array([[axes]])
    elif num_samples == 1:
        axes = np.expand_dims(axes, axis=0)
    elif len(valid_classes) == 1:
        axes = np.expand_dims(axes, axis=1)

    # Plotar imagens
    for col, cls in enumerate(valid_classes):
        for row, img in enumerate(class_images[cls]):
            ax = axes[row, col]

            if isinstance(img, Image.Image):
                img = to_tensor(img)

            if img.min() < 0:
                img = img * 0.5 + 0.5

            if img.ndimension() == 3:
                img = img.permute(1, 2, 0)

            ax.imshow(img.squeeze(), cmap='gray' if img.ndimension() == 2 or img.shape[-1] == 1 else None)
            ax.axis('off')

    # Ocultar quaisquer eixos vazios
    for ax in axes.flat:
        if len(ax.get_images()) == 0:
            ax.set_visible(False)

    plt.tight_layout()

    os.makedirs(save_path, exist_ok=True)
    path = os.path.join(save_path, "sample_images.png")
    # Salva o plot
    plt.savefig(path, bbox_inches='tight')  # bbox_inches evita corte de rótulos
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

def plot_original_data(dataset, train_data, test_data, path_dir):

    train_dist = get_label_distribution(train_data)
    test_dist = get_label_distribution(test_data)

    if dataset == "CIFAR-10":
        dataset = "CIFAR 10"
    elif dataset == "MNIST_FASHION":
        dataset = "MNIST-Fashion"
    elif dataset == "EMNIST_LETTERS":
        dataset = "EMNIST-Letters"


    title_train = f"Train Distribution - {dataset}"
    title_test = f"Test Distribution - {dataset}"
    plot_distribution(train_dist, "Original Train", path_dir, CLASS_COLORS, title_train)
    plot_distribution(test_dist, "Original Test", path_dir, CLASS_COLORS, title_test)


def plot_divided_data(dataset, full_set, train_set, val_set, test_set, path_dir):
    f"Train Distribution - {dataset}"
    full_dist = get_label_distribution(full_set)
    train_dist = get_label_distribution(train_set)
    val_dist = get_label_distribution(val_set)
    test_dist = get_label_distribution(test_set)

    if dataset == "CIFAR-10":
        dataset = "CIFAR 10"
    elif dataset == "MNIST_FASHION":
        dataset = "MNIST-Fashion"
    elif dataset == "EMNIST_LETTERS":
        dataset = "EMNIST-Letters"

    title_train = f"Train Distribution - {dataset}"
    title_val = f"Validation Distribution - {dataset}"
    title_test = f"Test Distribution - {dataset}"
    title_full = f"Full Class Distribution - {dataset}"
    
    save_class_distributions_to_csv(train_dist, val_dist, test_dist, path_dir)
    plot_distribution(train_dist, "Train", path_dir, CLASS_COLORS, title_train)
    plot_distribution(val_dist, "Validation", path_dir, CLASS_COLORS, title_val)
    plot_distribution(test_dist, "Test", path_dir, CLASS_COLORS, title_test)
    plot_distribution(full_dist, "Full", path_dir, CLASS_COLORS, title_full)

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


def stratified_split(dataset, train_pct=0.7, val_pct=0.1, test_pct=0.2):
    """
    Divide um dataset em subconjuntos de treino, validação e teste, garantindo que a distribuição das classes seja balanceada.
    """
    label_to_indices = defaultdict(list)

    # Agrupar índices por classe
    for idx, (_, label) in enumerate(dataset):
        label_to_indices[label].append(idx)

    train_indices, val_indices, test_indices = [], [], []

    for label, indices in label_to_indices.items():
        random.shuffle(indices)
        total = len(indices)
        n_train = int(train_pct * total)
        n_val = int(val_pct * total)
        n_test = total - n_train - n_val

        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train:n_train + n_val])
        test_indices.extend(indices[n_train + n_val:])

    return Subset(dataset, train_indices), Subset(dataset, val_indices), Subset(dataset, test_indices)