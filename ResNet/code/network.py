import torch
import torchvision
from torch import nn


class ShortcutBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, downsample_layer=None, stride=1):
        super(ShortcutBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample_layer
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        return self.relu(identity + x)


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, downsample_layer=None, stride=1):
        super(BottleneckBlock, self).__init__()
        features = int(out_channel * (64 / 64.)) * 1

        self.conv1 = nn.Conv2d(in_channel, features, 1)
        self.bn1 = nn.BatchNorm2d(features)
        self.conv2 = nn.Conv2d(features, features, 3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(features)
        self.conv3 = nn.Conv2d(features, out_channel * self.expansion, 1)
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        self.downsample = downsample_layer
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample_layer

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return self.relu(identity + x)


class ResNet(nn.Module):

    def __init__(self, num_classes, block, layers):
        super(ResNet, self).__init__()
        self.current_number_of_features = 64
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(3, stride=2)
        self.block1 = self._construct_layer(block, 64, layers[0])
        self.block2 = self._construct_layer(block, 128, layers[1], stride=2)
        self.block3 = self._construct_layer(block, 256, layers[2], stride=2)
        self.block4 = self._construct_layer(block, 512, layers[3], stride=2)
        self.pool2 = nn.AdaptiveAvgPool2d((1, 1))

        self.fc8 = nn.Linear(512 * block.expansion, num_classes)
        # self.actv8 = nn.Softmax(dim=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _construct_layer(self, block, output_channels, blocks, stride=1):
        downsample = None

        # If the input is going to be downsampled (stride = 2) or the number of
        #    channels is different from the expected expansion
        if stride != 1 or self.current_number_of_features != output_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.current_number_of_features, output_channels * block.expansion, 1, stride),
                nn.BatchNorm2d(output_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.current_number_of_features, output_channels, downsample, stride))
        self.current_number_of_features = output_channels * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.current_number_of_features, output_channels))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool2(x)

        x = torch.flatten(x, start_dim=1)
        x = self.fc8(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


