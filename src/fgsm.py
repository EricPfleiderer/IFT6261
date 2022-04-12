import torch
from torch import autograd
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

from src.trainable import TorchTrainable
class FGSM:

    def __init__(self, trainable: TorchTrainable, epsilon, targeted=True) -> None:
        self.trainable = trainable
        self.epsilon = epsilon
        self.targeted = targeted

    def fgsm(self, images, targets):
        images = images.clone().detach().to(self.trainable.device)

        loss_fn = nn.NLLLoss()
        images.requires_grad = True
        if self.targeted:
            outputs = -self.trainable(images)[:, targets]
            grad = torch.autograd.grad(outputs, images)[0]

        else:
            targets = targets.clone().detach().to(self.trainable.device)
            outputs = self.trainable.model(images)
            loss = loss_fn(outputs, targets)
            grad = torch.autograd.grad(loss, images)[0]

        adv_im = images + self.epsilon*grad.sign()
        adv_im = torch.clamp(adv_im, 0, 1)

        return adv_im

