from torch import nn


class MNIST(nn.Module):
    def __init__(self):
        super().__init__()

        self.hidden = nn.Linear(768, 256)
        self.output = nn.Linear(256, 10)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.softmax(x)

        return x


mnist = MNIST()

# print(mnist)

import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.hidden = nn.Linear(786, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        x = F.sigmoid(self.hidden(x))
        x = F.softmax(self.output(x))

        return x


network = Network()
print(network)
