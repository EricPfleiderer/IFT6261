import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src.loaders import get_dataset_by_name
from src.classifier import get_classifier_by_dataset_name
import logging

from src.gan import Discriminator, Generator

class Gan():
    def __init__(self, params, lr = 0.0002, g_input_dim=100, batch_size=100, epochs=200) -> None:
        self.params = params
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_loader, self.test_loader = get_dataset_by_name(params['dataset_name'], params['batch_size'])
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.g_input_dim = g_input_dim
        self.dataset_dim = self.train_loader.dataset.data.size(1)*self.train_loader.dataset.data.size(2)
        self.generator = Generator(self.g_input_dim, self.dataset_dim).to(self.device)
        self.discriminator = Discriminator(self.dataset_dim).to(self.device)
        self.criterion = nn.BCELoss()
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=self.lr)
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.lr)
   

    def d_train(self, x):

        self.discriminator.zero_grad()

        # Training on the real exemples
        x_real, y_real = x.view(-1, self.dataset_dim).to(self.device), torch.ones(self.batch_size, 1).to(self.device)
        output_real = self.discriminator(x_real)
        real_loss = self.criterion(output_real, y_real)

        # Training on the fake exemples
        z = torch.randn(self.batch_size, self.g_input_dim).to(self.device)
        x_fake, y_fake = self.generator(z), torch.zeros(self.batch_size, 1).to(self.device)
        output_fake = self.discriminator(x_fake)
        fake_loss = self.criterion(output_fake, y_fake)

        # Backpropagation
        loss = real_loss + fake_loss
        loss.backward()
        self.d_optimizer.step()

        return loss.data.item()


    def g_train(self, x):
        
        self.generator.zero_grad()

        z = torch.randn(self.batch_size, self.g_input_dim).to(self.device)
        y = torch.ones(self.batch_size, 1).to(self.device)

        g_output = self.generator(z)
        d_output = self.discriminator(g_output)
        g_loss = self.criterion(d_output, y)

        #Backpropagation
        g_loss.backward()
        self.g_optimizer.step()

        return g_loss.data.item()

    def training(self):
        for i in range(1, self.epochs+1):
            d_losses, g_losses = [], []
            for batch_idx, (x,_)in enumerate(self.train_loader):
                d_losses.append(self.d_train(x))
                g_losses.append(self.g_train(x))

            print('[%d/%d]: loss_d = %.3f, loss_g = %.3f' % ((epoch), self.epochs, torch.mean(torch.FloatTensor(d_losses)), torch.mean(torch.FloatTensor(g_losses))))
