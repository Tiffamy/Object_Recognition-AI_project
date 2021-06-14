import os
import torch
import torch.nn as nn
import torchvision.models as models
class vgg16(nn.Module):
    def vgg16(self):
        vgg16 = models.vgg16_bn()
        vgg16.classifier = nn.Sequential(nn.Linear(512, 10))
        vgg16.avgpool = nn.AvgPool2d((1, 1), stride=1)
        return vgg16