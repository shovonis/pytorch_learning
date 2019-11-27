import torch
import numpy as np


def activation(x):
    """ Sigmoid activation function

        Arguments
        ---------
        x: torch.Tensor
    """
    return 1 / (1 + torch.exp(-x))


torch.manual_seed(7)  # Set the random seed so things are predictable

# Features are 3 random normal variables
features = torch.randn((1, 3))

# Define the size of each layer in our network
n_input = features.shape[1]  # Number of input units, must match number of input features
n_hidden = 2  # Number of hidden units
n_output = 1  # Number of output units

W1 = torch.randn(n_input, n_hidden)
W2 = torch.randn(n_hidden, n_output)

# and bias terms for hidden and output layers
B1 = torch.randn((1, n_hidden))
B2 = torch.randn((1, n_output))

h = activation(torch.mm(features, W1) + B1)
output = activation(torch.mm(h, W2) + B2)

a = np.random.rand(4, 3)
b = torch.from_numpy(a)
print(b)
