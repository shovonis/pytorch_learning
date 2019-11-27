import torch
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
# Download and load the training data
trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.shape[0], -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)

        return x


#
# # Define the model
# model = Classifier()
# images, labels = next(iter(trainloader))

# # get the actual probability from log probability, because we used log soft max. We need to exp to get the probability
# logprob = model(images)
# ps = torch.exp(logprob)
#
# print(ps.shape)
#
# top_p, top_class = ps.topk(1, dim=1)
# # print(top_class)
# # print(top_p)
#
#
# equals = top_class == labels.view(*top_class.shape)
# print(equals.type(torch.FloatTensor))
# accuracy = torch.mean(equals.type(torch.FloatTensor))
# print(accuracy*100)
#


model = Classifier()

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

epochs = 30
step = 0
train_losses = []
test_losses = []

for epoch in range(epochs):
    running_loss = 0
    running_accuracy = 0

    for image, label in trainloader:
        optimizer.zero_grad()

        log_pred = model(image)
        loss = criterion(log_pred, label)
        loss.backward()
        optimizer.step()

        ps = torch.exp(log_pred)
        top_p, top_class = ps.topk(1, dim=1)
        equals = (top_class == label.view(*top_class.shape))
        running_accuracy += torch.mean(equals.type(torch.FloatTensor))

        running_loss += loss.item()

    else:
        test_loss = 0
        test_accuracy = 0

        with torch.no_grad():
            for images, labels in testloader:
                log_pred = model(images)
                test_loss += criterion(log_pred, labels)

                prob = torch.exp(log_pred)
                top_p, top_class = prob.topk(1, dim=1)
                equals = (top_class == labels.view(*top_class.shape))
                test_accuracy += torch.mean(equals.type(torch.FloatTensor))

        train_losses.append(running_loss / len(trainloader))
        test_losses.append(test_loss / len(testloader))

        print("Epoch: {}/{}.. ".format(epoch + 1, epochs),
              "Training Loss: {:.3f}.. ".format(running_loss / len(trainloader)),
              "Training Accuracy: {:.3f}.. ".format(running_accuracy / len(trainloader)),
              "Test Loss: {:.3f}.. ".format(test_loss / len(testloader)),
              "Test Accuracy: {:.3f}".format(test_accuracy / len(testloader)))
