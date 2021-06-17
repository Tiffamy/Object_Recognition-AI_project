import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as transforms

import numpy as np

from utils import *

models_opt = ["resnet18", "vgg16", "resnext29",
              "mobilenet", "mobilenet_v2", "googlenet", "densenet121"]

# define some arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="resnet18", type=str,
                    choices=models_opt, help="Which model you want to use.")
parser.add_argument("img_path", type=str, help="Please specific a image path!")

args = parser.parse_args()

# define best model path
model_path = "./pretrained/" + args.model + ".pth"

print("Read image from following path: {}".format(args.img_path))
# print(img2numpy(args.img_path).shape)

# check use cpu or gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("\nuse device: {}".format(device))


# load model
net = models(args.model)
net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)

best_model = torch.load(model_path)

net.load_state_dict(best_model['net'])
net.eval()

# define data transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Map predict value into corresponding class
classes = ('airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# load test image
test_img = img2numpy(args.img_path)
test_img_tensor = numpy_img_trans(test_img)
# print(test_img_tensor.shape)

# get prediction of every classes' probability
class_prob = net(test_img_tensor)

valur, pred = class_prob.max(1)
Class = classes[pred[0].item()]

print("The prediction of {} is {}".format(args.model, Class))
