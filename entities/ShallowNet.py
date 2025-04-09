# entities/ShallowNet.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ShallowNet(nn.Module):
    def __init__(self, num_classes=3):  # Ajusta o número de classes conforme necessário
        super(ShallowNet, self).__init__()

        # Uma única camada convolucional
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Como a entrada é 28x28, após pooling fica 14x14
        self.fc1 = nn.Linear(16 * 14 * 14, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [batch, 16, 14, 14]
        x = x.view(x.size(0), -1)             # Flatten: [batch, 16*14*14]
        x = self.fc1(x)                       # Sem softmax (CrossEntropyLoss cuida disso)
        return x
