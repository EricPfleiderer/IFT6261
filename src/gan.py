import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from src.loaders import get_dataset_by_name
from src.classifier import get_classifier_by_dataset_name
import logging


class Generator(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features*2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features*2)
        self.fc4 = nn.Linear(self.fc3.out_features, output_dim)
    
        def forward(self, x):
            x = F.leaky_relu(self.fc1(x), 0.2)
            x = F.leaky_relu(self.fc2(x), 0.2)
            x = F.leaky_relu(self.fc3(x), 0.2)

            return torch.tanh(self.fc4(x))


class Discriminator(nn.Module):
    def __init__(self, input_dim) -> None:
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features//2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features//2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)

    def forward(x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.droput(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)

        return torch.sigmoid(self.fc4(x))



