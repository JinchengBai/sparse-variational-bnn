# utility functions
import torch
import numpy as np


def sigmoid(z):
    return 1. / (1 + torch.exp(-z))


def logit(z):
    return torch.log(z/(1.-z))


def gumbel_softmax(logits, U, temperature, hard=False, eps=1e-20):
    """
        gumbel-softmax-approximation
    """
    z = logits + torch.log(U + eps) - torch.log(1 - U + eps)
    y = 1 / (1 + torch.exp(- z / temperature))
    if not hard:
        return y
    y_hard = (y > 0.5).double()
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard


def log_gaussian(x, mu, sigma):
    """
        log pdf of one-dimensional gaussian
    """
    if not torch.is_tensor(sigma):
        sigma = torch.tensor(sigma)
    return float(-0.5 * np.log(2 * np.pi)) - torch.log(sigma) - (x - mu)**2 / (2 * sigma**2)