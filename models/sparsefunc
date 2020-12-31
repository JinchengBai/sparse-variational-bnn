import torch
import torch.nn as nn
import numpy as np
from layers.layer import MLPLayer
from tools import log_gaussian


class SFunc(nn.Module):
    """
        Our BNN
    """
    def __init__(self, data_dim, device, target_dim=1, hidden_dim=7, sigma_noise=1.):

        # initialize the network using the MLP layer
        super(SFunc, self).__init__()
        self.rho_prior = torch.Tensor([np.log(np.exp(1.3) - 1)]).to(device)
        self.device = device

        self.l1 = MLPLayer(data_dim, hidden_dim, self.rho_prior, self.device)
        self.l1_relu = nn.ReLU()
        self.l2 = MLPLayer(hidden_dim, hidden_dim, self.rho_prior, self.device)
        self.l2_relu = nn.ReLU()
        self.l3 = MLPLayer(hidden_dim, hidden_dim, self.rho_prior, self.device)
        self.l3_relu = nn.ReLU()
        self.l4 = MLPLayer(hidden_dim, target_dim, self.rho_prior, self.device)

        self.target_dim = target_dim
        self.log_sigma_noise = torch.log(torch.Tensor([sigma_noise])).to(device)

    def forward(self, X, temp, phi_prior):
        """
            output of the BNN for one Monte Carlo sample

            :param X: [batch_size, data_dim]
            :return: [batch_size, target_dim]
        """
        output = self.l1_relu(self.l1(X, temp, phi_prior))
        output = self.l2_relu(self.l2(output, temp, phi_prior))
        output = self.l3_relu(self.l3(output, temp, phi_prior))
        output = self.l4(output, temp, phi_prior)
        return output

    def kl(self):
        # calculate the kl over all the layers of the BNN
        kl = self.l1.kl + self.l2.kl + self.l3.kl + self.l4.kl
        return kl

    def sample_elbo(self, X, y, n_samples, temp, phi_prior, num_batches):
        """
            calculate the loss function - negative elbo

            :param X: [batch_size, data_dim]
            :param y: [batch_size]
            :param n_samples: number of MC samples
            :param temp: temperature
            :return:
        """

        # initialization
        outputs = torch.zeros(n_samples, y.shape[0], self.target_dim).to(self.device)
        kls = 0.
        log_likes = 0.

        # make predictions and calculate prior, posterior, and likelihood for a given number of MC samples
        for i in range(n_samples):  # ith mc sample
            outputs[i] = self(X, temp, phi_prior)  # make predictions, (batch_size, target_dim)
            sample_kl = self.kl()  # get kl (a number)
            kls += sample_kl
            log_likes += torch.sum(log_gaussian(y, outputs[i].squeeze(), torch.exp(self.log_sigma_noise)))

        # calculate MC estimates of log prior, vb and likelihood
        kl_MC = kls/float(n_samples)
        # calculate negative loglikelihood
        nll_MC = - log_likes/float(n_samples)

        # calculate negative elbo
        loss = kl_MC / num_batches + nll_MC
        return loss, outputs.squeeze()
