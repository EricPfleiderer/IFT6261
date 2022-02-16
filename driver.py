import logging

import torch
import torch.optim as optim
import torch.nn.functional as F

import src.utils as utils
import src.loaders as loaders
from src.classifier import ClassifierMNIST


# Set stdout logging level
logging_level = logging.DEBUG
utils.set_logging(logging_level)

# Hyperparams
learning_rate = 0.001
momentum = 0.5
batch_size = 64
epochs = 20

# Data loaders
train_loader, test_loader = loaders.get_mnist_loaders(batch_size)

# MNIST classifier
classifier = ClassifierMNIST()
optimizer = optim.SGD(classifier.parameters(), lr=learning_rate, momentum=momentum)

# Training
logging.info(f'Training MNIST classifier for {epochs} epochs')
for epoch in range(epochs):

    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = classifier(data)
        train_loss = F.nll_loss(output, target)
        train_loss.backward()
        optimizer.step()

    test_loss = 0
    hit = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = classifier(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            hit += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(test_loader.dataset)

    logging.info(f'Epoch:{epoch}/{epochs}, train_loss:{round(train_loss.item(), 4)}, test_loss:{round(test_loss, 4)}')


# Apply genetic algorithm here!

