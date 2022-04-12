import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.loaders import get_dataset_by_name
from src.classifier import get_classifier_by_dataset_name
import logging

from src.trainable import TorchTrainable


class PGD():    
    def __init__(self, trainable : TorchTrainable, nb_steps=40, epsilon=0.3, alpha=2/255, targeted=True) -> None:
        self.trainable = trainable
        self.nb_steps = nb_steps
        self.epsilon = epsilon
        self.alpha = alpha
        self.targeted = targeted


    def pgd(self, images, targets):
        images = images.clone().detach().to(self.trainable.device)

        loss_fn = nn.CrossEntropyLoss()
        adv_images = images.clone().detach()

        for i in range(int(self.nb_steps)):
            adv_images.requires_grad = True
            if self.targeted :
                outputs = -self.trainable(adv_images)[:, targets]
                grad = torch.autograd.grad(outputs, adv_images)[0]
            else:
                targets = targets.clone().detach().to(self.trainable.device)
                outputs = self.trainable.model(adv_images)
                loss = loss_fn(outputs, targets)
                grad = torch.autograd.grad(loss, adv_images, retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.epsilon, max=self.epsilon)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
