import torch
import torch.nn as nn

class BiCLayer(torch.nn.Module):
    """
    Defining a BiC layer for a single task (2 parameters).
    """

    def __init__(self, numclass=10):
        super(BiCLayer, self).__init__()

        self.alpha = torch.nn.Parameter(torch.ones(numclass, requires_grad=True))
        self.beta = torch.nn.Parameter(torch.zeros(numclass, requires_grad=True))

    def forward(self, x):
        """
        Overloading the forward pass.
        """
        return self.alpha * x + self.beta