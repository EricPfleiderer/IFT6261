from src.hyperparams import *
import matplotlib.pyplot as plt


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