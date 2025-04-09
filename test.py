import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, ConcatDataset
import matplotlib.pyplot as plt
from utils.utils import *
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# LeNet-5 com Dropout e complexidade reduzida
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # Diminuir o número de filtros na primeira camada convolucional
        self.conv1 = nn.Conv2d(1, 4, kernel_size=5)  # Menos filtros
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=5)  # Menos filtros

        self.fc1 = nn.Linear(8 * 4 * 4, 60)  # Menos neurônios
        self.fc2 = nn.Linear(60, 40)         # Menos neurônios
        self.fc3 = nn.Linear(40, 10)         # 10 classes no FashionMNIST

        self.dropout1 = nn.Dropout(p=0.5)    # Maior dropout
        self.dropout2 = nn.Dropout(p=0.5)    # Maior dropout

    def forward(self, x):
        x = self.pool(torch.tanh(self.conv1(x)))
        x = self.pool(torch.tanh(self.conv2(x)))
        x = x.view(-1, 8 * 4 * 4)
        x = self.dropout1(torch.tanh(self.fc1(x)))
        x = self.dropout2(torch.tanh(self.fc2(x)))
        x = self.fc3(x)
        return x


# Função de treino com redução do número de épocas e porcentagem de treino (como decimal)
def train_model(model, train_loader, val_loader, num_epochs=10, lr=0.01, train_percentage=1.0):  # Agora o train_percentage é um decimal
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # Calcular o número de amostras a serem usadas no treino
    total_train_size = len(train_loader.dataset)
    train_size = int(train_percentage * total_train_size)  # Agora multiplicamos pelo decimal

    # Criar um subset com base no train_percentage
    indices = np.random.choice(total_train_size, train_size, replace=False)
    train_subset = Subset(train_loader.dataset, indices)
    train_subset_loader = DataLoader(train_subset, batch_size=train_loader.batch_size, shuffle=True)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_subset_loader:  # Usando o subset de treino
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_subset_loader)
        train_losses.append(avg_train_loss)

        # Avaliação em validação
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

    return train_losses, val_losses


# Avaliação final
def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"Test Accuracy: {acc:.2f}%")
    return acc

# Função para adicionar ruído aos dados de entrada
def add_noise_to_data(dataset, noise_factor=0.5):
    noisy_data = []
    for img, label in dataset:
        noisy_img = img + noise_factor * torch.randn_like(img)
        noisy_img = torch.clamp(noisy_img, 0., 1.)  # Limitar para [0,1]
        noisy_data.append((noisy_img, label))
    return noisy_data

# Main
def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Carregar e combinar FashionMNIST (train + test)
    train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    full_dataset = ConcatDataset([train_data, test_data])

    # Adicionar ruído aos dados para dificultar o aprendizado
    noisy_dataset = add_noise_to_data(full_dataset, noise_factor=0.5)

    # Cálculo das proporções: 70% treino, 20% teste, 10% validação
    total_size = len(noisy_dataset)
    train_size = int(0.7 * total_size)
    test_size = int(0.2 * total_size)
    val_size = total_size - train_size - test_size

    train_set, test_set, val_set = random_split(noisy_dataset, [train_size, test_size, val_size])

    # DataLoaders
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    model = LeNet5().to(device)

    print("Starting training...")
    train_losses, val_losses = train_model(model, train_loader, val_loader, num_epochs=3, lr=0.01)

    print("Evaluating on test set...")
    evaluate(model, test_loader)


    # Obter distribuições
    train_dist = get_label_distribution(train_set)
    val_dist = get_label_distribution(val_set)
    test_dist = get_label_distribution(test_set)
    print(f"Train, Val, Tes distribution: {train_dist}, {val_dist}, {test_dist}")


    
    # Plotar com múltiplas cores por barra
    plot_distribution(train_dist, "Train", colors=CLASS_COLORS)
    plot_distribution(val_dist, "Validation", colors=CLASS_COLORS)
    plot_distribution(test_dist, "Test", colors=CLASS_COLORS)

    # Plot das perdas
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_plot.png")
    plt.show()

if __name__ == "__main__":
    main()
