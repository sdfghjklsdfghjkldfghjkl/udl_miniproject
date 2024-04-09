import torch.nn as nn
import torch.nn.functional as F

class BaselineModel(nn.Module):
    """
    A baseline model for MNIST with two hidden layers and a configurable output layer.
    It's VI trained on full dataset.
    """
    
    def __init__(self, output_size):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(28 * 28, 256)  # First hidden layer
        self.dense2 = nn.Linear(256, 256)      # Second hidden layer
        self.classifier = nn.Linear(256, output_size)  # Output layer

    def forward(self, x, task_id):

        x = self.flatten(x)                    # Flatten the input
        x = F.relu(self.dense1(x))            # Activation function after first layer
        x = F.relu(self.dense2(x))            # Activation function after second layer
        return self.classifier(x)           # Output layer
        