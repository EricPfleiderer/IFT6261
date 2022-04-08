
import matplotlib.pyplot as plt
from src.configs import *

from src.trainable import TorchTrainable
from src.hyperparams import *
from src.GeneticAttack import *

import pickle
import shutil

import re
import torch


def generate_experiment_recap(path_to_experiment):

    import os
    import cv2

    # Fetch genetic attack object
    ga: GeneticAttack = pickle.load(open(path_to_experiment+'/models/ga.json', 'rb'))

    # Create temporary dir

    if os.path.isdir(path_to_experiment + '/temp'):
        shutil.rmtree(path_to_experiment + '/temp')

    os.makedirs(path_to_experiment + '/temp')

    # Generate frames from ga object
    figure, axis = plt.subplots()
    for epoch in range(ga.epochs):

        x = range(epoch)

        figure, axis = plt.subplots(2, 2)

        axis[0, 0].imshow(ga.history['best_solution'][epoch])
        axis[0, 0].set_title('Current solution')

        axis[0, 1].bar(range(0, 10), ga.history['prediction_dist'][epoch][0])
        axis[0, 1].set_title('Model predictions')
        axis[0, 1].set_ylim([0, 1])

        axis[1, 0].plot(x, ga.history['uncertainty_loss'][0:epoch])
        axis[1, 0].set_title('Uncertainty loss')
        axis[1, 0].set_xlim([0, ga.epochs])
        axis[1, 0].set_ylim([0, 1])

        axis[1, 1].plot(x, ga.epsilon*np.array(ga.history['sameness_loss'][0:epoch]))
        axis[1, 1].set_title('Sameness loss')
        axis[1, 1].set_xlim([0, ga.epochs])
        axis[1, 1].set_ylim([0, ga.epsilon*np.max(np.array(ga.history['sameness_loss']))])

        plt.savefig(path_to_experiment + '/temp/' + f'{epoch}.png')

    # Generate video from frames
    image_folder = path_to_experiment + '/temp'
    video_name = path_to_experiment + '/recap.avi'

    def natural_sort(l):
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(l, key=alphanum_key)

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images = natural_sort(images)

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 10, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

    shutil.rmtree(path_to_experiment+'/temp')


def find_candidates(trainable, threshold=0.25, trained=True):

    '''
    Given a trainable, this function finds potential targets for adversarial attacks.
    We compute the difference between the probability of the predicted class and all others.
    Candidates that have a difference higher than 'threshold' are chosen.
    :param trainable: .
    :param threshold:
    :param trained:
    :return:
    '''

    if not trained:
        trainable.train()

    nb_candidates = 0
    for data, target in trainable.test_loader:
        torch_predictions = trainable(data.to(trainable.device))
        np_predictions = np.exp(torch_predictions.cpu().detach().numpy())

        minus_max = np.max(np_predictions, axis=0) - np_predictions

        nb_candidates += np.count_nonzero(minus_max < threshold)

    return nb_candidates


def plot_candidates_vs_threshold(epochs=(5, 10, 15, 20), thresholds=np.arange(0, 1, 0.05)):

    '''
    Trains a trainable for every number of epochs and computes the number of candidates as a function of thresholds.

    :param epochs:
    :param thresholds:
    :return:
    '''

    space = CNN_space.copy()
    params = []

    # Defining training and model parameters
    for epoch in epochs:
        space['num_epochs'] = epoch
        params.append(dict(**space, **configs))

    # Trainable class
    trainables = [TorchTrainable(param) for param in params]
    for trainable in trainables:
        trainable.train()

    results = []
    for trainable in trainables:
        outputs = []
        for threshold in thresholds:
            outputs.append(find_candidates(trainable, threshold))
        results.append(outputs)

    plt.figure()
    plt.title('Potential number of candidates as a function of threshold')

    for idx, value in enumerate(epochs):
        plt.plot(thresholds, results[idx], label=f'{value} epochs')
    plt.legend(loc='best')
    plt.savefig('outputs/candidates_vs_threshold.jpg')