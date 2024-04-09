import torch.nn as nn
import torch.nn.functional as F

from layers import BaysLinear
from layers import BaysConv


class MultiheadModelConvolutional(nn.Module):
    """
    Defines a Multi head with Bayesian layers, used for tasks with separate classifiers.
    """
    
    def __init__(self, input_size, num_layers=2, hidden_size=150, num_tasks=5):
        """
        Initializes the model with two Bayesian hidden layers and five classifiers.
        """
        super().__init__()
        out_chan = 16
        self.lower_nets = nn.ModuleList([
            BaysConv(3, out_chan, 3), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            BaysConv(out_chan,out_chan, 3), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            BaysLinear(576, hidden_size), nn.ReLU()
        ])

        self.head_nets = nn.ModuleList([BaysLinear(hidden_size, 2) for _ in range(num_tasks)])


    def forward(self, x, task_id):
        out = x
        for ln in self.lower_nets:
            out = ln(out)
        return self.head_nets[task_id](out)

    def get_kl(self, task_id):
        """ Computes the total KL divergence for the specified task. """
        kl = 0
        for ln in self.lower_nets:
            if isinstance(ln, BaysLinear) or isinstance(ln, BaysConv):
                kl = kl + ln.kl_loss()
        kl = kl + self.head_nets[task_id].kl_loss()
        return kl

    def update_prior(self):
        """ Updates prior distributions for all Bayesian layers. """
        for layer in self.children():
            if layer.__class__ is not nn.ModuleList:
                layer.prior_weight_m = layer.weight_m.data
                layer.prior_weight_v = layer.weight_v.data
                
    def reset_parameters(self):
        for layer in self.children():
            if layer.__class__ is not nn.ModuleList:
                layer.reset_parameters()


