import logging

import torch
import numpy as np
import matplotlib.pyplot as plt

import src.utils as utils
from src.trainable import TorchTrainable
from src.hyperparams import *
from src.configs import configs


# Set stdout logging level
logging_level = logging.INFO
utils.set_logging(logging_level)


trainable = TorchTrainable(dict(**CNN_space, **configs))
trainable.train()
trainable.plot_history()

candidates = [trainable.test_loader.dataset.data[0].cpu().detach()]



# Attacker algorithm here
def whitebox_ga_attack(trainable, population_size=10):

    # 1. Choose attack candidates based on model certainty
    # 2. Perform a genetic algorithm attack on the candidates based on a loss function including model certainty
    pass




# Attacker algorithm here
def blackbox_ga_attack(trainable, population_size=10):

    # 1. Choose attack candidates at random
    # 2. Perform a genetic algorithm attack on the candidates by using only their prediction
    pass
