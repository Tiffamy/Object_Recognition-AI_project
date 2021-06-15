import os
import torch
import torch.nn as nn
import torchvision.models as model

def vgg16():
    vgg16 = model.vgg16_bn()

    vgg16.classifier = nn.Sequential(nn.Linear(512, 10))
    vgg16.avgpool = nn.AvgPool2d((1, 1), stride=1)

    return vgg16
