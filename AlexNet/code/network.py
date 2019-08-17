import torch
import torchvision
from torch import nn



class AlexNet(nn.Module):

    def __init__(self,num_classes):
        super(AlexNet, self).__init__()

        self.conv1 = nn.Conv2d(3,96,11,stride=4,padding=2)
        self.actv1 = nn.ReLU()
        # self.normalisation = nn.LocalResponseNorm()  ## Not used anymore
        self.pool1 = nn.MaxPool2d(3,2)

        self.conv2 = nn.Conv2d(96,256,5,stride=1,padding=2)
        self.actv2 = nn.ReLU()
        # self.normalisation = nn.LocalResponseNorm()  ## Not used anymore
        self.pool2 = nn.MaxPool2d(3,2)

        self.conv3 = nn.Conv2d(256,384,3,stride=1,padding=1)
        self.actv3 = nn.ReLU()

        self.conv4 = nn.Conv2d(384, 384, 3, stride=1, padding=1)
        self.actv4 = nn.ReLU()

        self.conv5 = nn.Conv2d(384, 256, 3, stride=1, padding=1)
        self.actv5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(3,2)

        self.fc6 = nn.Linear(6*6*256,4096)
        self.actv6 = nn.ReLU()
        self.dropout6 = nn.Dropout(0.5)

        self.fc7 = nn.Linear(4096, 4096)
        self.actv7 = nn.ReLU()
        self.dropout7 = nn.Dropout(0.5)

        self.fc8 = nn.Linear(4096,num_classes)
        self.actv8 = nn.Softmax()


    def forward(self, x):

        x = self.actv1(self.conv1(x))
        x = self.pool1(x)

        x = self.actv2(self.conv2(x))
        x = self.pool2(x)

        x = self.actv3(self.conv3(x))

        x = self.actv4(self.conv4(x))

        x = self.actv5(self.conv5(x))
        x = self.pool5(x)

        x = x.reshape(-1)
        x = self.dropout6(self.actv6(self.fc6(x)))

        x = self.dropout7(self.actv7(self.fc7(x)))

        x = self.actv8(self.fc8(x))

        return x

