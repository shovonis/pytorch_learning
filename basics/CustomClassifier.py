from torch import nn


class CustomClassfier(nn.Module):
    def __init__(self):
        super().__init__()

        self.hidden = nn.Linear(768, 128)
        self.second_hidden = nn.Linear(128, 64)
        self.output = nn.Linear(64, 10)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.second_hidden(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.softmax(x)

        return x


import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 128)
        self.fc2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.output(x))

        return x


network = Network()
print(network)

input_size = 784
input_sizes = [128, 64]
output_size = 10

model = nn.Sequential(nn.Linear(input_size, input_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(input_sizes[0], input_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(input_sizes[1], output_size),
                      nn.Softmax(dim=1))

print(model)
