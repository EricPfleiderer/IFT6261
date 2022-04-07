import logging
import pickle

import matplotlib.pyplot as plt

import torch
import src.utils as utils
from src.GeneticAttack import GeneticAttack
from src.trainable import TorchTrainable
from src.hyperparams import *
from src.configs import configs


# Set stdout logging level
logging_level = logging.INFO
utils.set_logging(logging_level)

# Train, save and/or load TorchTrainable
# trainable = TorchTrainable(dict(**CNN_space, **configs))
# trainable.train()
# pickle.dump(trainable, open('models/test.json', 'wb'))
trainable = pickle.load(open('models/test.json', 'rb'))

# Attack sample (single out of sample image and target)
x, y = trainable.test_loader.dataset[788][0][0], trainable.test_loader.dataset[788][1]

ga = GeneticAttack(x, y, trainable, N=200, epochs=100, selective_pressure=0.4, asexual_repro=1, epsilon=0.05,
                   uncertainty_power=2, sameness_power=2)

original_prediction_dist = trainable(x)
print(f'The model thinks this is an image of a {torch.argmax(original_prediction_dist)} with a confidence of {torch.max(original_prediction_dist)}')

# Initial image
plt.figure()
plt.imshow(x)
plt.show()

adversarial_prediction_dist = trainable(ga.best_solution)
print(f'The model thinks this is an image of a {torch.argmax(adversarial_prediction_dist)} with a confidence of {torch.max(adversarial_prediction_dist)}')

# Adversarial image
plt.figure()
plt.imshow(ga.best_solution)
plt.show()

