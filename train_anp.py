import numpy as np
import torch

from math import pi
from random import randint
from torch import nn
from torch.distributions.kl import kl_divergence

from models import ANP
from utils import context_target_split


class NeuralProcessTrainer():
    """
    Class to handle training of Neural Processes for functions.
    Parameters
    ----------
    device : torch.device
    neural_process : NeuralProcess instance
    optimizer : one of torch.optim optimizers
    num_context_range : tuple of ints
        Number of context points will be sampled uniformly in the range given
        by num_context_range.
    num_extra_target_range : tuple of ints
        Number of extra target points (as we always include context points in
        target points, i.e. context points are a subset of target points) will
        be sampled uniformly in the range given by num_extra_target_range.
    print_freq : int
        Frequency with which to print loss information during training.
    """
    def __init__(self, device, neural_process, optimizer, num_context_range,
                 num_extra_target_range, print_freq=100):
        self.device = device
        self.neural_process = neural_process
        self.optimizer = optimizer
        self.num_context_range = num_context_range
        self.num_extra_target_range = num_extra_target_range
        self.print_freq = print_freq
        self.steps = 0
        self.epoch_loss_history = []

    def train(self, data_loader, epochs):
        """
        Trains Neural Process.
        Parameters
        ----------
        dataloader : torch.utils.DataLoader instance
        epochs : int
            Number of epochs to train for.
        """
        for epoch in range(epochs):
            epoch_loss = 0.
            for i, data in enumerate(data_loader):
                self.optimizer.zero_grad()

                # Sample number of context and target points
                num_context = randint(*self.num_context_range)
                num_extra_target = randint(*self.num_extra_target_range)

                x, y = data
                x_context, y_context, x_target, y_target = \
                    context_target_split(x, y, num_context, num_extra_target)
                p_y_pred, q_target, q_context = \
                    self.neural_process(x_context, y_context, x_target, y_target)

                loss = self._loss(p_y_pred, y_target, q_target, q_context)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

                self.steps += 1

                if self.steps % self.print_freq == 0:
                    print("iteration {}, loss {:.3f}".format(self.steps, loss.item()))

            print("Epoch: {}, Avg_loss: {}".format(epoch, epoch_loss / len(data_loader)))
            self.epoch_loss_history.append(epoch_loss / len(data_loader))

    def _loss(self, p_y_pred, y_target, q_target, q_context):
        """
        Computes Neural Process loss.
        Parameters
        ----------
        p_y_pred : one of torch.distributions.Distribution
            Distribution over y output by Neural Process.
        y_target : torch.Tensor
            Shape (batch_size, num_target, y_dim)
        q_target : one of torch.distributions.Distribution
            Latent distribution for target points.
        q_context : one of torch.distributions.Distribution
            Latent distribution for context points.
        """
        # Log likelihood has shape (batch_size, num_target, y_dim). Take mean
        # over batch and sum over number of targets and dimensions of y
        log_likelihood = p_y_pred.log_prob(y_target).mean(dim=0).sum()
        # KL has shape (batch_size, r_dim). Take mean over batch and sum over
        # r_dim (since r_dim is dimension of normal distribution)
        kl = kl_divergence(q_target, q_context).mean(dim=0).sum()
        return -log_likelihood + kl


if __name__ == '__main__':
    from datasets import SineData
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create dataset
    dataset = SineData(amplitude_range=(-1., 1.),
                       shift_range=(-.5, .5),
                       num_samples=2000)

    x_dim = 1
    y_dim = 1
    r_dim = 50  # Dimension of representation of context points
    z_dim = 50  # Dimension of sampled latent variable
    h_dim = 50  # Dimension of hidden layers in encoder and decoder

    neuralprocess = ANP(x_dim, y_dim, r_dim, z_dim, h_dim)

    batch_size = 2
    num_context = 4
    num_target = 4

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(neuralprocess.parameters(), lr=3e-4)
    np_trainer = NeuralProcessTrainer(device, neuralprocess, optimizer,
                                      num_context_range=(num_context, num_context),
                                      num_extra_target_range=(num_target, num_target),
                                      print_freq=200)

    neuralprocess.training = True
    np_trainer.train(data_loader, 30)
