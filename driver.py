import logging
from src.trainable import TorchTrainable
from src.hyperparams import CNN_space
from src.configs import configs

import src.utils as utils


# Set stdout logging level
logging_level = logging.DEBUG
utils.set_logging(logging_level)

# Defining training and model parameters
params = dict(**CNN_space, **configs)

# Trainable class
trainable = TorchTrainable(params)
trainable.train()


