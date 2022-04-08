import os
import pickle
import logging
import csv
from datetime import datetime

import torch

import src.utils as utils
from src.GeneticAttack import GeneticAttack
from src.trainable import TorchTrainable
from src.hyperparams import CNN_space, GA_space
from src.configs import configs
from src.graphing import *

# Set stdout logging level
logging_level = logging.INFO
utils.set_logging(logging_level)


def run_experiment(x_index=788, trained_classifier_path=None, root='outputs/experiments/'):

    now = datetime.now()
    full_path = root + 'Experiment_' + str(now) + '/'  # Full path to current experiment
    os.makedirs(full_path)

    # Train from scratch
    if trained_classifier_path is None:
        params = dict(**CNN_space, **configs)
        trainable = TorchTrainable(params)
        trainable.train()
        os.makedirs(full_path+'models/')
        pickle.dump(trainable, open(full_path+'models/' + 'classifier.json', 'wb'))

        # Save classifier params to csv
        w = csv.writer(open(full_path + 'classifier_params.csv', "w"))
        for key, val in params.items():
            w.writerow([key, val])

    # Load from save
    else:
        trainable = pickle.load(open(trained_classifier_path, 'rb'))

    # Graph classifier history
    trainable.plot_history(path=full_path)

    # Attack sample (single out of sample image and target)
    x, y = trainable.test_loader.dataset[x_index][0][0], trainable.test_loader.dataset[x_index][1]

    params = GA_space
    ga = GeneticAttack(x, y, trainable, **params)

    # Graph the genetic attack history
    ga.plot_history(path=full_path)
    pickle.dump(ga, open(full_path + 'models/' + 'ga.json', 'wb'))

    # Save genetic attack params to csv
    w = csv.writer(open(full_path + 'ga_params.csv', "w"))
    for key, val in params.items():
        w.writerow([key, val])

    original_prediction_dist = trainable(x)
    print(f'The model thinks this is an image of a {torch.argmax(original_prediction_dist)} with a confidence of {torch.max(original_prediction_dist)}')

    adversarial_prediction_dist = trainable(ga.best_solution)
    print(f'The model thinks this is an image of a {torch.argmax(adversarial_prediction_dist)} with a confidence of {torch.max(adversarial_prediction_dist)}')

    generate_experiment_recap(full_path)


run_experiment()
