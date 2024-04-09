import torch.nn as nn
import torch.nn.functional as F

from layers import BaysLinear

class MultiheadModel(nn.Module):
    """
    Defines a Multi head with Bayesian layers, used for tasks with separate classifiers.
    """
    
    def __init__(self, input_size, num_layers=2, hidden_size=256, num_tasks=5):
        """
        Initializes the model with two Bayesian hidden layers and five classifiers.
        """
        super().__init__()
        self.lower_nets = nn.ModuleList([BaysLinear(input_size, hidden_size)])
        for _ in range(num_layers-1):
            self.lower_nets.append(BaysLinear(hidden_size, hidden_size))
        self.head_nets = nn.ModuleList([BaysLinear(hidden_size, 2) for _ in range(num_tasks)])

    def forward(self, x, task_id):
        out = x.view(x.size(0), -1)
        for ln in self.lower_nets:
            out = F.relu(ln(out))
        return self.head_nets[task_id](out)

    def get_kl(self, task_id):
        """ Computes the total KL divergence for the specified task. """
        kl = 0
        for ln in self.lower_nets:
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


