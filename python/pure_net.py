from __future__ import print_function

from torch import nn
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from python.common_net import register_weight_layer
from python.timer_utils import NamedTimerInstance, VerboseLevel

print("CUDA is available:", torch.cuda.is_available())
torch.backends.cudnn.deterministic = True

dtype = torch.float
device = torch.device("cpu")

minibatch = 512
n_classes = 10  # Create random Tensors to hold inputs and outputs
target = torch.randn(minibatch, n_classes)


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
     ])

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.Compose([
    transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4), transforms.ToTensor(), normalize, ]))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=minibatch, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.Compose([
    transforms.ToTensor(), normalize, ]))
testloader = torch.utils.data.DataLoader(testset, batch_size=minibatch, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        need_bias = False
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=need_bias)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1, bias=need_bias)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1, bias=need_bias)
        self.conv4 = nn.Conv2d(256, 256, 3, padding=1, bias=need_bias)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(256, 512, 3, padding=1, bias=need_bias)
        self.conv6 = nn.Conv2d(512, 512, 3, padding=1, bias=need_bias)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv7 = nn.Conv2d(512, 512, 3, padding=1, bias=need_bias)
        self.conv8 = nn.Conv2d(512, 512, 3, padding=1, bias=need_bias)
        self.pool = nn.MaxPool2d(2, 2)
        # self.NumElemFlatten = int(262144 / minibatch)
        self.NumElemFlatten = 512
        self.fc1 = nn.Linear(self.NumElemFlatten, 512)
        self.fc2 = nn.Linear(512, 512, bias=need_bias)
        self.fc3 = nn.Linear(512, 10, bias=need_bias)
        self.norm1 = nn.BatchNorm2d(64)
        self.norm2 = nn.BatchNorm2d(128)
        self.norm3 = nn.BatchNorm2d(256)
        self.norm4 = nn.BatchNorm2d(256)
        self.norm5 = nn.BatchNorm2d(512)
        self.norm6 = nn.BatchNorm2d(512)
        self.norm7 = nn.BatchNorm2d(512)
        self.norm8 = nn.BatchNorm2d(512)
        self.normfc1 = nn.BatchNorm1d(512)
        self.normfc2 = nn.BatchNorm1d(512)

    def forward(self, x):
        x = self.pool(F.relu(self.norm1(self.conv1(x))))
        x = self.pool(F.relu(self.norm2(self.conv2(x))))
        x = self.pool(F.relu(self.norm4(self.conv4(F.relu(self.norm3(self.conv3(x)))))))
        x = self.pool(F.relu(self.norm6(self.conv6(F.relu(self.norm5(self.conv5(x)))))))
        x = self.pool(F.relu(self.norm8(self.conv8(F.relu(self.norm7(self.conv7(x)))))))
        x = x.view(-1, self.NumElemFlatten)
        x = F.relu(self.normfc1(self.fc1(x)))
        x = F.relu(self.normfc2(self.fc2(x)))
        x = self.fc3(x)
        return x


net = Net()
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

register_weight_layer(net.conv1, "conv1")
register_weight_layer(net.conv2, "conv2")
register_weight_layer(net.conv3, "conv3")
register_weight_layer(net.conv4, "conv4")
register_weight_layer(net.conv5, "conv5")
register_weight_layer(net.conv6, "conv6")
register_weight_layer(net.conv7, "conv7")
register_weight_layer(net.conv8, "conv8")
# register_linear_layer(net.fc1, "fc1")

NumShowInter = 100
NumEpoch = 200
torch.set_num_threads(1)
# https://github.com/chengyangfu/pytorch-vgg-cifar10
for epoch in range(NumEpoch):  # loop over the dataset multiple times
    scheduler.step()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        with NamedTimerInstance("TrainWithBatch", VerboseLevel.RUN):
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        if i % NumShowInter == NumShowInter - 1 or (epoch == 0 and i == 0):  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / NumShowInter))
            running_loss = 0.032
            correct = 0
            total = 0

            # store_layer_name = 'conv8'
            # store_name = f"{store_layer_name}_{epoch + 1}_{i + 1}"
            # store_layer(store_layer_name, store_name)

            with torch.no_grad():
                for data in testloader:
                    images, labels = data[0].to(device), data[1].to(device)
                    outputs = net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

print('Finished Training')
