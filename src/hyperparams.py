import numpy as np
import torch.optim as optim

CNN_space = {
    'batch_size': 64,
    'num_epochs': 5,
    'optimizer': {
        'type': optim.SGD,  # Must be callable
        'opt_params': {
            'lr': 0.001,
            'momentum': 0.5,
        },
    }
}

GA_space = {
    'epochs': 100,
    'N': 30,
    'selective_pressure': 0.4,
    'asexual_repro': 1.0,
    'epsilon': 0.05,
    'uncertainty_power': 2,
    'sameness_power': 2,
    'mutation_size': 0.025,
}