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
    def __init__(self, device='cpu', dataset=DATASET_CIFAR_10, epochs=EPHOCS, lr=LEARNING_RATE, batch_size=64):
        super(LeNet5, self).__init__()
        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size

        if (dataset == DATASET_CIFAR_10): 
            in_channels = 3
        else:
            in_channels = 1
        
         # Camada 1: Convolucional com 6 filtros de 5x5
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5)
        self.pool = nn.AvgPool2d(2, 2)  # Usando average pooling como na LeNet original
        # Camada 2: Convolucional com 16 filtros de 5x5
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        

        if dataset == DATASET_CIFAR_10:
            # Após conv1 (6 filtros 5x5) + pool: 28x28 -> 14x14
            # Após conv2 (16 filtros 5x5) + pool: 10x10 -> 5x5
            fc_input_size = 16 * 5 * 5  # 16 filtros, cada um de tamanho 5x5 após pooling
        else:
             # Para MNIST, a entrada é 28x28 e a estrutura é similar.
            fc_input_size = 16 * 4 * 4  # Após as camadas convolucionais e pooling

        self.fc1 = nn.Linear(fc_input_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, len(CLASSES))
        self.dropout = nn.Dropout(0.5)  # Dropout com probabilidade de 50% por padrão
        self.to(self.device)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 + ReLU + Pooling
        x = self.pool(F.relu(self.conv2(x)))  # Conv2 + ReLU + Pooling
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Aplica o Dropout após a primeira camada fully connected
        x = F.relu(self.fc2(x))
        x = self.dropout(x)  # Aplica o Dropout após a segunda camada fully connected
        x = self.fc3(x)
        return x

    def init_fit(self, X, y, epochs=INIT_EPHOCS):
        self.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        if isinstance(X, torch.Tensor):
            X_tensor = X.clone().detach().float()
        else:
            raise TypeError("Input X must be a torch tensor")

        
        if X_tensor.dim() != 4 and X_tensor.dim() != 3:
             raise ValueError(f"Unexpected input shape: {X_tensor.shape}\nInput X must have at least 3 dimensions (C, H, W).")

        if isinstance(y, torch.Tensor):
            y_tensor = y.clone().detach().long()
        else:
            raise TypeError("Input y must be a torch tensor")


        dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(epochs):
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
            print(f"    Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f}")

        return self

    def fit(self, X, y):
        self.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        if isinstance(X, torch.Tensor):
            X_tensor = X.clone().detach().float()
        else:
            raise TypeError("Input X must be a torch tensor")

        
        if X_tensor.dim() != 4 and X_tensor.dim() != 3:
             raise ValueError(f"Unexpected input shape: {X_tensor.shape}\nInput X must have at least 3 dimensions (C, H, W).")

        if isinstance(y, torch.Tensor):
            y_tensor = y.clone().detach().long()
        else:
            raise TypeError("Input y must be a torch tensor")


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

        # Classes únicas
        classes = sorted(np.unique(y_tensor.cpu().numpy()))  

        # Confusion Matrix
        cm = confusion_matrix(all_labels, all_preds, labels=classes)

        # Accuracy por classe
        acc_per_class = []
        for i in range(len(cm)):
            correct = cm[i, i]
            total = cm[i, :].sum()
            acc = correct / total if total > 0 else 0
            acc_per_class.append(float(acc))

        # Precision, Recall, F1 Score por classe
        precision_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0, labels=classes).tolist()
        recall_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0, labels=classes).tolist()
        f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0, labels=classes).tolist()

        # Specificity por classe
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
            "sensitivity_per_class": [f"{r:.4f}" for r in recall_per_class],  # Sensitivity é o mesmo que recall
            "specificity_per_class": [f"{s:.4f}" for s in specificity_per_class],
            "confusion_matrix": cm.tolist()
        }

        return formatted_metrics

