import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# Load CIFAR 10 dataset

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Define network

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # todo
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.maxp1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.maxp2 = nn.MaxPool2d(2, stride=2)

        self.fc1 = nn.Linear(400, 120, bias=True)
        self.fc2 = nn.Linear(120, 84, bias=True)
        self.fc3 = nn.Linear(84, 10, bias=True)

    def forward(self, x):
        # todo
        out = F.relu(self.conv1(x))
        out = self.maxp1(out)
        out = F.relu(self.conv2(out))
        out = self.maxp2(out)

        out = out.view(-1, 400);
        
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        return self.fc3(out)

# todo: copy network to cuda
net = Net()

# Define cross entropy loss and SGD optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(list(lr=0.001, momentum=0.9))

for epoch in range(2):  
    train_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        # todo: copy data to cuda
        train_input = Variable(inputs.cuda()) 
        train_label = Variable(labels.cuda())

        # todo: zero the parameter gradients
        optimizer.zero_grad()

        # todo forward + backward + optimize
        train_out = net.forward(train_input)
        loss = criterion(train_out, train_label)
        loss.backward()
        optimizer.step()

        # print statistics
        train_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, train_loss / 2000))
            train_loss = 0.0

print('Finished Training')