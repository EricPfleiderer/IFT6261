import logging
import torch.utils.data as data
import torchvision.datasets as datasets

import torchvision.transforms as transforms


# MNIST
def get_mnist_loaders(batch_size, root='data/', download=True, shuffle=True):

    logging.info('Preparing MNIST data loaders')
    train_loader = data.DataLoader(datasets.MNIST(root, train=True, download=download, transform=transforms.ToTensor()), batch_size=batch_size, shuffle=shuffle)
    test_loader = data.DataLoader(datasets.MNIST(root, train=False, download=download, transform=transforms.ToTensor()), batch_size=batch_size, shuffle=shuffle)
    logging.debug(f'Train data shape: {train_loader.dataset.data.shape}')
    logging.debug(f'Test data shape: {test_loader.dataset.data.shape}')

    # TODO: apply transforms to data

    return  train_loader, test_loader

