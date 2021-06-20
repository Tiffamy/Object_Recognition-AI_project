import os

import torch
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import cv2 as cv

import numpy as np
from PIL import Image

from old_models.resnet18 import *
from old_models.vgg16 import *
from old_models.densenet import *
from old_models.googlenet import *
from old_models.mobilenet import *
from old_models.mobilenetv2 import *
from old_models.resnext import *

from our_models.our_model import *
from our_models.our_model_v2 import *
from our_models.our_model_v3 import *
from our_models.our_model_v4 import *
# In oredr to load the numpy dataset and tranform to type which fit pytorch


class MyDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = torch.LongTensor(targets)
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        if self.transform:
            x = Image.fromarray(self.data[index].astype(np.uint8))
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)


# Convert image into numpy array
def img2numpy(path):
    # load image and convert into 32x32 RGB image
    image = cv.imread(path)
    img = cv.resize(image, (32, 32), interpolation=cv.INTER_AREA)
    img_RGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    image_np = np.array(img_RGB)

    return image_np


# load specific model
def models(model):
    if model == "resnet18":
        return ResNet18()
    if model == "vgg16":
        return vgg16()
    if model == "resnext29":
        return ResNeXt29_2x64d()
    if model == "mobilenet":
        return MobileNet()
    if model == "mobilenet_v2":
        return MobileNetV2()
    if model == "googlenet":
        return GoogLeNet()
    if model == "densenet121":
        return DenseNet121()
    if model == "our_model":
        return our_model()
    if model == "our_model_v2":
        return our_model_v2()
    if model == "our_model_v3":
        return our_model_v3()
    if model == "our_model_v4":
        return our_model_v4()


# define data transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


# convert img numpy array into tensor and normalize it
def numpy_img_trans(img_np):
    # convert to PIL image
    img = torch.zeros((1, 3, 32, 32))
    image = Image.fromarray(np.uint8(img_np))
    image = transform(image)

    img[0] = image

    return img
