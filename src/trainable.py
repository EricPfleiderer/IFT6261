from src.loaders import get_dataset_by_name
from src.classifier import get_classifier_by_dataset_name
import logging

import torch
import torch.nn.functional as F


class TorchTrainable:

    def __init__(self, params):

        self.params = params
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_loader, self.test_loader = get_dataset_by_name(params['dataset_name'], params['batch_size'])

        self.model = get_classifier_by_dataset_name(params['dataset_name']).to(self.device)

        if 'optimizer' in params:
            opt_callable = params['optimizer']['type']
            self.optimizer = opt_callable(self.model.parameters(), **params['optimizer']['opt_params'])

    def train(self):
        logging.info(f'Training MNIST classifier for ' + str(self.params['num_epochs']) + ' epochs')
        for epoch in range(self.params['num_epochs']):

            # Training
            train_loss = 0
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                train_loss = F.nll_loss(output, target)
                train_loss.backward()
                self.optimizer.step()

            # Validation
            test_loss = 0
            hit = 0
            with torch.no_grad():
                for data, target in self.test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    test_loss += F.nll_loss(output, target, size_average=False).item()
                    predictions = output.data.max(1, keepdim=True)[1]
                    hit += predictions.eq(target.data.view_as(predictions)).sum()
                test_loss /= len(self.test_loader.dataset)

            logging.info(f'Epoch: {epoch}/' + str(self.params['num_epochs']) +
                         ', train loss: ' + '{:.4f}'.format(round(train_loss.item(), 4)) +
                         ', test_loss: ' + '{:.4f}'.format(round(test_loss, 4)))
   

    def infer(self, x):
        self.model.eval()
        if len(x.shape) == 2:
            # If x is a single image, unsqueeze twice (once for channel, once for batch)
            x = torch.unsqueeze(x, dim=0)
            x = torch.unsqueeze(x, dim=0)

        elif len(x.shape) == 3:
            # If x is of dim 3, it is assumed that the missing dim is the channel dim.
            x = torch.unsqueeze(x, dim=1)

        preds = torch.exp(self.model(x.to(self.device)))

        return preds

    def __call__(self, x):
        return self.infer(x)
