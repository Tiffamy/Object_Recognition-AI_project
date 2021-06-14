import os
import torch
import torch.nn as nn
import torch.nn.functional as F


import numpy as np

# define basic block


class ResidualBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# define ResNet18
class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # stage1
        self.layer1 = nn.Sequential(
            ResidualBlock(64, 64, 1),
            ResidualBlock(64, 64, 1)
        )

        # stage2
        self.layer2 = nn.Sequential(
            ResidualBlock(64, 128, 2),
            ResidualBlock(128, 128, 1)
        )

        # stage3
        self.layer3 = nn.Sequential(
            ResidualBlock(128, 256, 2),
            ResidualBlock(256, 256, 1)
        )

        # stage4
        self.layer4 = nn.Sequential(
            ResidualBlock(256, 512, 2),
            ResidualBlock(512, 512, 1)
        )

        # fully connected layer
        self.linear = nn.Linear(512, num_classes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
