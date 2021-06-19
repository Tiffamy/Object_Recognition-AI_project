import os
import torch
import torch.nn as nn
import torch.nn.functional as F


import numpy as np

class ResidualBlock_v2(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(ResidualBlock_v2, self).__init__()
    
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
class our_model(nn.Module):
    def __init__(self, num_classes=10):
        super(our_model, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        
        # stage1
        self.layer1 = nn.Sequential(
            ResidualBlock_v2(64, 64, 1),
            ResidualBlock_v2(64, 64, 1)
        )

        # stage2
        self.layer2 = nn.Sequential(
            ResidualBlock_v2(64, 128, 2),
            ResidualBlock_v2(128, 128, 1)
        )

        self.shortcut1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1,
                      stride=2, bias=False),
            nn.BatchNorm2d(128)
        )

        # stage3
        self.layer3 = nn.Sequential(
            ResidualBlock_v2(128, 256, 2),
            ResidualBlock_v2(256, 256, 1)
        )

        self.shortcut2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1,
                      stride=2, bias=False),
            nn.BatchNorm2d(256)
        )

        # stage4
        self.layer4 = nn.Sequential(
            ResidualBlock_v2(256, 512, 2),
            ResidualBlock_v2(512, 512, 1)
        )

        self.shortcut3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1,
                      stride=2, bias=False),
            nn.BatchNorm2d(512)
        )

        # fully connected layer
        self.linear = nn.Linear(512, num_classes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        
        rec = out
        out = self.layer1(out)
        out += rec
        rec = out
        out = self.layer2(out)
        out += self.shortcut1(rec)
        rec = out
        out = self.layer3(out)
        out += self.shortcut2(rec)
        rec = out
        out = self.layer4(out)
        out += self.shortcut3(rec)
        
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
