import torch
from torch import nn



class VGG(nn.Module):
    def __init__(self):
        super(VGG,self).__init__()

        self.conv1_1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.pool1 = nn.MaxPool2d(stride=2,kernel_size=2)

        self.conv2_1 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1)
        self.pool2 = nn.MaxPool2d(stride=2,kernel_size=2)

        self.conv3_1 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1)
        self.conv3_3 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1)
        self.pool3 = nn.MaxPool2d(stride=2,kernel_size=2)

        self.conv4_1 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1)
        self.conv4_2 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1)
        self.conv4_3 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1)
        self.pool4 = nn.MaxPool2d(stride=2,kernel_size=2)

        self.conv5_1 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1)
        self.conv5_2 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1)
        self.conv5_3 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1)
        self.pool5 = nn.MaxPool2d(stride=2,kernel_size=2)

        self.fc6 = nn.Linear(7*7*512,4096)
        self.fc7 = nn.Linear(4096,4096)
        self.fc8 = nn.Linear(4096,1000)

    def forward(self, x):


        return x