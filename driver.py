import os
import pickle
import logging
import json
from datetime import datetime

import torch

import src.utils as utils
from src.GeneticAttack import GeneticAttack
from src.trainable import TorchTrainable
from src.hyperparams import *
from src.configs import configs

# Set stdout logging level
logging_level = logging.INFO
utils.set_logging(logging_level)


def run_experiment(x_index=788, trained_classifier_path=None, root='outputs/experiments/'):

    # Train from scratch
    if trained_classifier_path is None:
        params = dict(**CNN_space, **configs)
        trainable = TorchTrainable(params)
        trainable.train()
        os.makedirs(root+'models/')
        pickle.dump(trainable, open(root+'models/' + 'classifier.json', 'wb'))
        pickle.dump(params, open(root+'models/' + 'classifier_params.json', 'wb'))

    # Load from save
    else:
        trainable = pickle.load(open(trained_classifier_path, 'rb'))

    now = datetime.now()
    full_path = root + 'Experiment_' + str(now) + '/'
    os.makedirs(full_path)

    trainable.plot_history(path=full_path)

    # Attack sample (single out of sample image and target)
    x, y = trainable.test_loader.dataset[x_index][0][0], trainable.test_loader.dataset[x_index][1]

    ga = GeneticAttack(x, y, trainable, N=500, epochs=100, selective_pressure=0.4, asexual_repro=1, epsilon=0.05,
                       uncertainty_power=2, sameness_power=2)
    ga.plot_history(path=full_path)
    pickle.dump(ga, open(root + 'models/' + 'ga.json', 'wb'))

    original_prediction_dist = trainable(x)
    print(f'The model thinks this is an image of a {torch.argmax(original_prediction_dist)} with a confidence of {torch.max(original_prediction_dist)}')

    adversarial_prediction_dist = trainable(ga.best_solution)
    print(f'The model thinks this is an image of a {torch.argmax(adversarial_prediction_dist)} with a confidence of {torch.max(adversarial_prediction_dist)}')


run_experiment()
