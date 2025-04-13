import torch
import torch.nn as nn
import os
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from collections import Counter
import torch.nn.functional as F
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix,
    accuracy_score, roc_auc_score, roc_curve, log_loss
)
from sklearn.preprocessing import label_binarize
from env import *

class LeNet5(nn.Module):
    def __init__(self, device='cpu', epochs=INIT_TRAINING_EPHOCHS, lr=INIT_LEARNING_RATE, batch_size=64):
        super(LeNet5, self).__init__()
        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        
        # Reduzindo a capacidade:
        # Agora, conv1 tem 2 filtros e conv2 tem 4 filtros.
        self.conv1 = nn.Conv2d(1, 2, kernel_size=5)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(2, 4, kernel_size=5)
        
        # Cálculo do tamanho de entrada da primeira camada FC:
        # Com imagens 28x28:
        # Após conv1 (kernel=5): 28-5+1 = 24 -> tamanho 24x24, e após pool: 12x12.
        # Após conv2 (kernel=5): 12-5+1 = 8 -> tamanho 8x8, e após pool: 4x4.
        # Com 4 filtros, a saída é 4 * 4 * 4 = 64.
        self.fc1 = nn.Linear(4 * 4 * 4, 30)  # Reduzido
        self.fc2 = nn.Linear(30, 20)         # Reduzido
        self.fc3 = nn.Linear(20, 10)         # 10 classes no FashionMNIST
        
        # Aumentando o dropout para ajudar a desacelerar o aprendizado
        self.dropout1 = nn.Dropout(p=0.6)
        self.dropout2 = nn.Dropout(p=0.6)
        
        # Mover o modelo para o dispositivo configurado
        self.to(self.device)

    def forward(self, x):
        # Aplicando as camadas com tanh e pooling
        x = self.pool(torch.tanh(self.conv1(x)))
        x = self.pool(torch.tanh(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Achatar o tensor para a camada FC
        x = self.dropout1(torch.tanh(self.fc1(x)))
        x = self.dropout2(torch.tanh(self.fc2(x)))
        x = self.fc3(x)
        return x

    # Função de treinamento (fit)
    def fit(self, X, y):
        self.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-3)

        if isinstance(X, np.ndarray):
            X_tensor = torch.from_numpy(X).float()
        elif isinstance(X, torch.Tensor):
            X_tensor = X.clone().detach().float()
        else:
            raise TypeError("Input X must be a numpy array or torch tensor")

        y_numpy = y.numpy() if isinstance(y, torch.Tensor) else y

        if X_tensor.dim() == 3:
            X_tensor = X_tensor.unsqueeze(0)
        if X_tensor.dim() == 4 and X_tensor.shape[1] != 1:
            X_tensor = X_tensor.unsqueeze(1)

        if isinstance(y, np.ndarray):
            y_tensor = torch.from_numpy(y).long()
        elif isinstance(y, torch.Tensor):
            y_tensor = y.clone().detach().long()
        else:
            raise TypeError("Input y must be a numpy array or torch tensor")

        if y_tensor.dim() == 0:
            y_tensor = y_tensor.unsqueeze(0)

        dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_train_loss = running_loss / len(train_loader)
            print(f"    Epoch [{epoch+1}/{self.epochs}] - Train Loss: {avg_train_loss:.4f}")

        return self

    # Função de previsão (predict)
    def predict(self, X):
        self.eval()  
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(self.device)  
            outputs = self(X_tensor)  
            _, predicted = torch.max(outputs, 1)
        return predicted.cpu().numpy()

    def predict_proba(self, X):
        self.eval()  # Coloca o modelo no modo de avaliação
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X_tensor = torch.from_numpy(X).float()
            elif isinstance(X, torch.Tensor):
                X_tensor = X.clone().detach().float()
            else:
                raise TypeError("Input X must be a numpy array or torch tensor")
            
            if X_tensor.dim() == 3:  # [batch_size, 28, 28]
                X_tensor = X_tensor.unsqueeze(1)  # Vira [batch_size, 1, 28, 28]
            
            X_tensor = X_tensor.to(self.device)
            outputs = self(X_tensor)
            probabilities = F.softmax(outputs, dim=1)
        
        return probabilities.cpu().numpy()

    def evaluate(self, X, y, batch_size=64):
        self.eval()
        self.to(self.device)

        # Garantir que X e y sejam tensores
        if isinstance(X, np.ndarray):
            X_tensor = torch.from_numpy(X).float()
        elif isinstance(X, torch.Tensor):
            X_tensor = X.clone().detach().float()
        else:
            raise TypeError("Input X must be a numpy array or torch tensor")

        if isinstance(y, np.ndarray):
            y_tensor = torch.from_numpy(y).long()
        elif isinstance(y, torch.Tensor):
            y_tensor = y.clone().detach().long()
        else:
            raise TypeError("Input y must be a numpy array or torch tensor")

        # DataLoader temporário
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self(images)
                probs = F.softmax(outputs, dim=1)
                _, preds = torch.max(probs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        classes = sorted(np.unique(y_tensor.cpu().numpy()))  # ou simplesmente classes = [0, 1, 2] se souberes de antemão

        # Métricas por classe (corrigido)
        precision_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0, labels=classes).tolist()
        recall_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0, labels=classes).tolist()
        f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0, labels=classes).tolist()

        # Confusion Matrix (corrigido)
        cm = confusion_matrix(all_labels, all_preds, labels=classes)

        # Accuracy por classe
        acc_per_class = []
        for i in range(len(cm)):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - (tp + fn + fp)
            acc = (tp + tn) / cm.sum() if cm.sum() > 0 else 0
            acc_per_class.append(float(acc))

        # Specificity
        specificity_per_class = []
        for i in range(len(cm)):
            tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
            fp = cm[:, i].sum() - cm[i, i]
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            specificity_per_class.append(float(specificity))

        # Resultado final formatado
        formatted_metrics = {
            "accuracy_per_class": [f"{acc:.4f}" for acc in acc_per_class],
            "precision_per_class": [f"{p:.4f}" for p in precision_per_class],
            "recall_per_class": [f"{r:.4f}" for r in recall_per_class],
            "f1_score_per_class": [f"{f1:.4f}" for f1 in f1_per_class],
            "sensitivity_per_class": [f"{r:.4f}" for r in recall_per_class],
            "specificity_per_class": [f"{s:.4f}" for s in specificity_per_class],
            "confusion_matrix": cm.tolist()
        }

        return formatted_metrics




# === Define the device (CPU ou CUDA) ===
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# === Create Model Instance ===
model = LeNet5(device=device)

# === Create Output Directory ===
output_dir = "models"
os.makedirs(output_dir, exist_ok=True)

# === Define Save Path ===
save_path = os.path.join(output_dir, "lenet5_modified_base.pth")

# === Save Model Weights (untrained) ===
torch.save(model.state_dict(), save_path)
print(f"lenet5_modified model saved to: {save_path}")