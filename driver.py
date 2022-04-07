import logging
import pickle

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

ga = GeneticAttack(x, y, trainable)


