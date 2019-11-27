import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch import optim

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_data = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

# Defining the model
model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1))

# The loss function
criterion = nn.NLLLoss()

# Data loader
images, labels = next(iter(trainloader))

# Flattening the images
images = images.view(images.shape[0], -1)

# predictions
logits = model(images)

# Calculate Loss
loss = criterion(logits, labels)

# Optimizers require the parameters to optimize and a learning rate
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Back Propagation
loss.backward()
print('Gradient -', model[0].weight.grad)

# Weight update through optimizer
optimizer.step()
print('Updated weights - ', model[0].weight)

