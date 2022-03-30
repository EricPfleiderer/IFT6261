import logging

import torch

import src.utils as utils
from src.trainable import TorchTrainable
from src.hyperparams import *
from src.configs import configs


# Set stdout logging level
logging_level = logging.INFO
utils.set_logging(logging_level)

# Attack target
trainable = TorchTrainable(dict(**CNN_space, **configs))
trainable.train()
trainable.plot_history()

# Attack sample (single out of sample image and target)
x, y = trainable.test_loader.dataset[777][0][0], trainable.test_loader.dataset[777][1]


# Attacker algorithm here
def whitebox_ga_attack(x: torch.Tensor, y: int, trainable: TorchTrainable, population_size=10, epochs=10, elite_perc=0.4, epsilon=1, uncertainty_power=2, sameness_power=2):
    """
    :param x: 28 by 28 torch tensor of the original image.
    :param y: Target.
    :param trainable: Trainable targeted by the attack.
    :param population_size; Size of the population during the simulation.
    :param epochs: Number of epochs to run the genetic algorithm.
    :param elite_perc: Percentage of the most fit population that are considered during reproduction.
    :param epsilon: Coefficient for the linear combination between uncertainty and sameness in the loss.
    :param uncertainty_power: Power of the exponent used in the uncertainty equation.
    :param sameness_power: Power of the exponent used in the sameness equation.
    :return:
    """

    # Create a population
    population = torch.stack([x for i in range(population_size)])

    def evaluate_quality(x: torch.Tensor, y: int, adversarial_x, trainable: TorchTrainable, epsilon=1,
                         uncertainty_power=2, sameness_power=2):

        """
        :param x: 28 by 28 torch tensor of the original image.
        :param y: Target.
        :param adversarial_x: batch of 28 by 28 perturbed images
        :param trainable: Trainable targeted by the attack.
        :param epsilon: Coefficient for the linear combination between uncertainty and sameness in the loss.
        :param uncertainty_power: Power of the exponent used in the uncertainty equation.
        :param sameness_power: Power of the exponent used in the sameness equation.
        :return:
        """
        # TODO: Find optimal parameters for quality eval (epsilon, powers)

        adversarial_prediction = torch.exp(trainable(torch.unsqueeze(adversarial_x, dim=1)))
        adversarial_prediction[:, y] = 0
        loss = torch.sum(torch.pow(adversarial_prediction, uncertainty_power), dim=0) # TODO: REVISIT LOSS TO ENSURE LOSS IS DRIVEN TOWARDS A SINGLE OTHER CLASS
        return loss + epsilon * torch.sum((x-adversarial_x)**sameness_power)

    for i in range(epochs):

        # Perturb the population
        population += torch.normal(0, 0.05, size=population.shape)  # TODO: implement perturbation decay
        population = torch.clamp(population, 0, 1)

        # Evaluate the quality of the population
        qual = evaluate_quality(x, y, population, trainable)

        # Choose the fittest units for reproduction (population_size/2 parents chosen with replacement among the fitest)


        # Create the new generation from the fittest parents


        # Elitism (maintain top solution at all times)
        pass

    return


whitebox_ga_attack(x, y, trainable)


# # Attacker algorithm here
# def blackbox_ga_attack(trainable, population_size=10):
#
#     # 1. Choose attack candidates at random
#     # 2. Perform a genetic algorithm attack on the candidates by using only their prediction
#     pass
