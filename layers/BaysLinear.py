import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

class BaysLinear(nn.Module):

    def __init__(self, input_dim, out_features, priors=None):
        """
        Initializes the Bayesian linear layer with specified input/output dimensions and priors.

        Args:
            input_dim (int): Number of input features.
            out_features (int): Number of output features.
            priors (dict): Prior distributions for the weights.
        """
        super().__init__()
        self.input_dim = input_dim
        self.out_features = out_features

        # Setting default priors if none provided
        default_priors = {'mu': 0, 'sigma': 0.01}
        self.priors = priors or default_priors

        # Initializing prior parameters
        self.prior_weight_m = torch.tensor(self.priors['mu'])
        self.prior_weight_v = torch.tensor(self.priors['sigma'])

        # Initializing parameters for mean (m) and standard deviation (v)
        self.weight_m = Parameter(torch.Tensor(out_features, input_dim))
        self.weight_v = Parameter(torch.Tensor(out_features, input_dim))

        # Initializing bias and weight parameters
        self.bias = Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        """
        Resets parameters to initial values.
        """
        self.weight_m.data.normal_(0, 0.1)
        self.weight_v.data.fill_(-3) #initializes the log varinace parameter to -3
        self.bias.data.normal_(0, 0.1)
        
        self.prior_weight_m = torch.tensor(self.priors['mu'])
        self.prior_weight_v = torch.tensor(self.priors['sigma'])

    def forward(self, x, sample=True):
        """
        Forward pass with the option to sample from the posterior.

        Args:
            x (torch.Tensor): The input tensor.
            sample (bool): Whether to sample from the posterior distribution.
        """
        # Transforming v to sigma
        weight_v = torch.log1p(torch.exp(self.weight_v))

        # Calculating mean and variance of activations
        activation_m = F.linear(x, self.weight_m, self.bias)
        activation_var = 1e-16 + F.linear(x.pow(2), weight_v.pow(2))#, bias_variance)
        activation_std = torch.sqrt(activation_var)

        # Sampling from Gaussian if in training mode or sampling is requested
        if self.training or sample:
            epsilon = torch.randn_like(activation_m).to(x.device)
            return activation_m + activation_std * epsilon
        return activation_m

    def kl_loss(self):
        # Calculating KL divergence for weights
        kl_divergence = KL(self.prior_weight_m, self.prior_weight_v, self.weight_m, self.weight_v)
        return kl_divergence

# class FlattenLayer(nn.Module):
    
#     def __init__(self, num_features):
#         super().__init__()
#         self.num_features = num_features

#     def forward(self, x):
#         return x.view(-1, self.num_features)


def KL(mu_p, sig_p, mu_q, sig_q):
    kl = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).sum()
    return kl