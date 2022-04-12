import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import matplotlib.pyplot as plt
import os
import pickle
import csv
import torch
import src.utils as utils
import numpy as np

from datetime import datetime
from src.trainable import TorchTrainable
from src.hyperparams import CNN_space
from src.configs import configs
from src.fgsm import FGSM

# Set stdout logging level
logging_level = logging.INFO
utils.set_logging(logging_level)

def run_fgsm_target(epsilon, x_index=788, trained_classifier_path=None, root='outputs/experiments/'):

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
    print(x.shape, y)
    fgsm = FGSM(trainable, epsilon=epsilon)

    adv_im = fgsm.fgsm(x, y)
    original_prediction = trainable(x)
    perturb_prediction = trainable(adv_im)
    print(f'The model thinks this is an image of a {torch.argmax(original_prediction)} with a confidence of {torch.max(original_prediction)}')
    print(f'The model thinks this is an image of a {torch.argmax(perturb_prediction)} with a confidence of {torch.max(perturb_prediction)}')

    #Plot true exemple and adversarial exemple
    adv_im = adv_im.detach().cpu().numpy()


    return torch.argmax(original_prediction).item(), torch.argmax(perturb_prediction).item(), torch.max(perturb_prediction).item(), adv_im

def run_fgsm(epsilon, trained_classifier_path=None, root='outputs/experiments'):

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

    adv_examples = []
    correct = 0
    fgsm = FGSM(trainable, epsilon, targeted=False)
    for _, (x, y) in enumerate(trainable.test_loader):
        adv_x = fgsm.fgsm(x, y)
        original_prediction = trainable(x)
        perturb_prediction = trainable(adv_x)
        for i in range(len(x)):
            if torch.argmax(perturb_prediction[i]).item() == y[i].item():
                correct += 1
                if (epsilon == 0.0) and (len(adv_examples) < 5):
                    adv_ex = adv_x[i].squeeze().detach().cpu().numpy()
                    adv_examples.append((torch.argmax(original_prediction[i]).item(), torch.argmax(perturb_prediction[i]).item(), torch.max(perturb_prediction[i]).item(), adv_ex))
            else :
                if len(adv_examples) < 5:
                    adv_ex = adv_x[i].squeeze().detach().cpu().numpy()
                    adv_examples.append((torch.argmax(original_prediction[i]).item(), torch.argmax(perturb_prediction[i]).item(), torch.max(perturb_prediction[i]).item(), adv_ex))

            # print(f'The model thinks this is an image of a {torch.argmax(original_prediction[i])} with a confidence of {torch.max(original_prediction[i])}')
            # print(f'The model thinks this is an image of a {torch.argmax(perturb_prediction[i])} with a confidence of {torch.max(perturb_prediction[i])}')

    final_acc = correct/(len(trainable.test_loader)*64)
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(trainable.test_loader), final_acc))
    return final_acc, adv_examples

epsilons = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

############# Target FGSM #############
adv_images = []
for eps in epsilons:
    orig, pert, pred, adv_ex = run_fgsm_target(epsilon=eps, trained_classifier_path='outputs/experiments/Experiment_2022-04-09 20:11:20.763245/models/classifier.json')
    adv_images.append((orig, pert, pred, adv_ex))

print(len(adv_images))
plt.figure(figsize=(8, 10))
count=0
for i in range(len(epsilons)):
    count += 1
    plt.subplot(len(epsilons), 2, count)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.ylabel("Eps : {}".format(epsilons[i]))
    plt.xlabel("Certainty : {}".format(round(adv_images[i][2], 2)))
    plt.title("{} -> {}".format(adv_images[i][0], adv_images[i][1]))
    plt.imshow(adv_images[i][3], cmap='gray')

plt.tight_layout()
plt.show()



############# Untargeted FGSM ############
adv_images_untargeted = []
accuracies = []
for eps in epsilons:
    acc, adv_ex = run_fgsm(epsilon=eps, trained_classifier_path='outputs/experiments/Experiment_2022-04-09 20:11:20.763245/models/classifier.json')
    adv_images_untargeted.append(adv_ex)
    accuracies.append(acc)
print(accuracies)

#Accuracies plot
plt.figure(figsize=(5,5))
plt.plot(epsilons, accuracies)
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(0, 0.35, step=0.05))
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.title("Accuracy in function of epsilon value")
plt.show()

#Adversarial examples plot
cnt = 0
plt.figure(figsize=(8,10))
for i in range(len(epsilons)):
    for j in range(len(adv_images_untargeted[i])):
        cnt += 1
        plt.subplot(len(epsilons),len(adv_images_untargeted[0]),cnt)
        orig,adv,pred,ex = adv_images_untargeted[i][j]
        plt.xticks([], [])
        plt.yticks([], [])
        if j == 0:
            plt.ylabel("Eps: {}".format(epsilons[i]))
        plt.xlabel("Certainty : {}".format(round(pred, 2)), fontsize=10)
        plt.title("{} -> {}".format(orig, adv))
        plt.imshow(ex, cmap="gray")
plt.tight_layout()
plt.show()
