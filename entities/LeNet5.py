import torch
import torch.nn as nn
import os

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

# === Create Model Instance ===
model = LeNet5()

# === Create Output Directory ===
output_dir = "models"
os.makedirs(output_dir, exist_ok=True)

# === Define Save Path ===
save_path = os.path.join(output_dir, "lenet5_modified_base.pth")

# === Save Model Weights (untrained) ===
torch.save(model.state_dict(), save_path)
print(f"lenet5_modified model saved to: {save_path}")
