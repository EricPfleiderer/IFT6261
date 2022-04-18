import numpy as np
import torch.optim as optim

CNN_space = {
    'batch_size': 64,
    'num_epochs': 20,
    'optimizer': {
        'type': optim.SGD,  # Must be callable
        'opt_params': {
            'lr': 0.001,
            'momentum': 0.5,
        },
    }
}
GAN_space = {'batch_size': 100}

