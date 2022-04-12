import os
import pickle
import logging
import csv
from datetime import datetime

import torch

import src.utils as utils
from src.trainable import TorchTrainable
from src.hyperparams import CNN_space
from src.configs import configs

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

    x, y = trainable.test_loader.dataset[x_index][0][0], trainable.test_loader.dataset[x_index][1]

run_experiment()
