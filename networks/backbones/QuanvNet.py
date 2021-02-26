import torch.nn as nn
import torch
import pennylane as qml
from pennylane import numpy as np
# import pennylane_qulacs

from networks.backbones.custom_layers.QuanvLayer import Quanv


class QuanvNet(nn.Module):
    def __init__(self):
        super(QuanvNet, self).__init__()
        # Conv Layers
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv3 = nn.Conv2d(20, 5, kernel_size=1)

        self.convALT = nn.Conv2d(5, 4, kernel_size=2) # Acts like the Quanv Layer

        # Quanvolution Layer
        self.Quanv1 = Quanv(kernal_size=3, output_depth=4, circuit_layers=1)
        self.AdaptPool = nn.AdaptiveMaxPool2d(3)

        # Fully Connected Layers
        self.fc1 = nn.Linear(36, 12)
        self.fc2 = nn.Linear(12, 3)

        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        # x = self.sigmoid(self.convALT(x))
        x = self.sigmoid(self.Quanv1(x))
        x = self.AdaptPool(x)
        x = x.view(-1, 36)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
