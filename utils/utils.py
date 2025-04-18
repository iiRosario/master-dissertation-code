import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Subset
from collections import Counter
from env import *
import csv
import pandas as pd
from torchvision.transforms import functional as F
from matplotlib.ticker import MaxNLocator


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

""" # Função para plotar 3 figuras do dataset para as classes 0, 1 e 2
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
    plt.show() """


def plot_sample_images(dataset, classes=[0, 1, 2], num_samples=3):
    # Mapear as classes numéricas para seus respectivos nomes no FashionMNIST ou CIFAR-10
    class_names = ['T-shirt/top', 'Trouser', 'Pullover']  # Para FashionMNIST (adaptar para CIFAR-10)

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
            # Verificar se a imagem é RGB ou em escala de cinza
            if img.ndimension() == 3:  # RGB (C, H, W)
                img = img.permute(1, 2, 0)  # Transforma para (H, W, C)
            
            axes[i, j].imshow(img.squeeze(), cmap='gray' if img.ndimension() == 2 else None)
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
    plot_metric_over_cycles(csv_path=csv_path, 
                            plot_path=plot_path, 
                            variable="accuracy_per_class",
                            filename=f"accuracy_{seed}")
    
    plot_metric_over_cycles(csv_path=csv_path, 
                            plot_path=plot_path, 
                            variable="precision_per_class",
                            filename=f"precision_{seed}")



def after_run_plot_metric(csv_folder, plot_path, variable, filename):
    all_metrics = {}

    # Ler todos os CSVs da pasta
    for csv_file in os.listdir(csv_folder):
        if csv_file.endswith(".csv"):
            csv_path = os.path.join(csv_folder, csv_file)
            df = pd.read_csv(csv_path)

            # Converter a coluna da métrica de string para lista
            def parse_list(row):
                return [float(x) for x in row.strip("[]").replace("'", "").split(',')]

            df[variable] = df[variable].apply(parse_list)

            # Agregar métricas por ciclo
            for _, row in df.iterrows():
                cycle = int(row['cycle'])
                values = row[variable]

                if cycle not in all_metrics:
                    all_metrics[cycle] = []
                all_metrics[cycle].extend(values)

    # Ordenar ciclos e calcular média + std
    sorted_cycles = sorted(all_metrics.keys())
    avg_per_cycle = [np.mean(all_metrics[cycle]) for cycle in sorted_cycles]
    std_per_cycle = [np.std(all_metrics[cycle]) for cycle in sorted_cycles]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_cycles, avg_per_cycle, label=f'Average {variable}', color='blue')
    plt.fill_between(
        sorted_cycles,
        np.array(avg_per_cycle) - np.array(std_per_cycle),
        np.array(avg_per_cycle) + np.array(std_per_cycle),
        color='blue',
        alpha=0.2,
        label='±1 STD'
    )

    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.title(f'Average {variable.replace("_", " ").title()} over Active Learning Cycles')
    plt.xlabel('Cycle')
    plt.ylabel(variable.replace("_", " ").title())
    plt.legend()
    plt.grid(True)

    os.makedirs(plot_path, exist_ok=True)
    plot_file = os.path.join(plot_path, f'{filename}.png')
    plt.savefig(plot_file)
    plt.close()

    print(f"Plot saved to: {plot_file}")


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