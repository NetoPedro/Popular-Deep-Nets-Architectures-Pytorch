import torch
from torch import nn



class VGG16(nn.Module):
    def __init__(self,num_classes):
        super(VGG16, self).__init__()

        self.conv1_1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.actv1_1 = nn.ReLU()
        self.conv1_2 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.actv1_2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(stride=2,kernel_size=2)

        self.conv2_1 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1)
        self.actv2_1 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1)
        self.actv2_2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(stride=2,kernel_size=2)

        self.conv3_1 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1)
        self.actv3_1 = nn.ReLU()
        self.conv3_2 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1)
        self.actv3_2 = nn.ReLU()
        self.conv3_3 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1)
        self.actv3_3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(stride=2,kernel_size=2)

        self.conv4_1 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1)
        self.actv4_1 = nn.ReLU()
        self.conv4_2 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1)
        self.actv4_2 = nn.ReLU()
        self.conv4_3 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1)
        self.actv4_3 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(stride=2,kernel_size=2)

        self.conv5_1 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1)
        self.actv5_1 = nn.ReLU()
        self.conv5_2 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1)
        self.actv5_2 = nn.ReLU()
        self.conv5_3 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1)
        self.actv5_3 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(stride=2,kernel_size=2)

        self.fc6 = nn.Linear(7*7*512,4096)
        self.actv6 = nn.ReLU()
        self.dropout6 = nn.Dropout(0.5)

        self.fc7 = nn.Linear(4096,4096)
        self.actv7 = nn.ReLU()
        self.dropout7 = nn.Dropout(0.5)

        self.fc8 = nn.Linear(4096,num_classes)

    def forward(self, x):

        x = self.actv1_1(self.conv1_1(x))
        x = self.actv1_2(self.conv1_2(x))
        x = self.pool1(x)

        x = self.actv2_1(self.conv2_1(x))
        x = self.actv2_2(self.conv2_2(x))
        x = self.pool2(x)

        x = self.actv3_1(self.conv3_1(x))
        x = self.actv3_2(self.conv3_2(x))
        x = self.actv3_3(self.conv3_3(x))
        x = self.pool3(x)

        x = self.actv4_1(self.conv4_1(x))
        x = self.actv4_2(self.conv4_2(x))
        x = self.actv4_3(self.conv4_3(x))
        x = self.pool4(x)

        x = self.actv5_1(self.conv5_1(x))
        x = self.actv5_2(self.conv5_2(x))
        x = self.actv5_3(self.conv5_3(x))
        x = self.pool5(x)

        x = torch.flatten(x, start_dim=1)

        x = self.dropout6(self.actv6(self.fc6(x)))

        x = self.dropout7(self.actv7(self.fc7(x)))

        x = self.fc8(x)
        return x