import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import time
import numpy as np
import random

# Simple feedforward neural network
class SimpleNN(nn.Module):
    def __init__(self, activation):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 10) # 2 input features, 10 hidden units
        self.fc2 = nn.Linear(10, 1) # 10 hidden units, 1 output unit
        self.activation = activation # Custom activation function

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # Output layer for binary classification
        return x


