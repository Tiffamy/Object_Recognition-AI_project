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
import matplotlib.pyplot as plt

from utils import *
from autoAug import AutoAugment, Cutout
# import new_models

models_opt = ["resnet18", "vgg16", "resnext29",
              "mobilenet", "mobilenet_v2", "googlenet", "densenet121", "our_model", "our_model_v2", "our_model_v3", "our_model_v4"]

# set some arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="resnet18", type=str,
                    choices=models_opt, help="Which model you want to use.")
parser.add_argument("--lr", default=0.1, type=float, help="learning rate.")
parser.add_argument("--epoch", default=200, type=int,
                    help="Train how many epochs.")
parser.add_argument("--train_batch", default=128,
                    type=int, help="training batch size.")
parser.add_argument(
    "--pretrained", help="Use pretrained model or not.", action="store_true")

args = parser.parse_args()

# print(args.model, args.lr, args.train_batch, args.pretrained)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("use device: {}".format(device))
# rec best test accuracy
best_acc = 0

out_path = './pretrained/' + args.model + '.pth'

# define dataset transform
transform_train = [
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
]

if args.model == "our_model_v3" or args.model == "our_model_v4":
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ])

    transform_train.append(AutoAugment())
    transform_train.append(Cutout())

else:
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

transform_train.extend([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

transform_train = transforms.Compose(transform_train)


# load dataset


"""
# to categorical, e,g, [1] -> [0, 1, 0, 0.....]
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

didn't use
"""


def to_categorical(y, num_classed):
    return np.eye(num_classed, dtype='uint8')[y]


trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.train_batch, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

if not args.pretrained:
    # Declare model
    net = models(args.model)

    net = net.to(device)

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=200)

    best_acc = 0
    # Training loop
    for epoch in range(args.epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        correct = 0
        total = 0

        # loop over all the batch
        for _, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            value, pred = outputs.max(1)
            total += targets.size(0)
            correct += pred.eq(targets).sum().item()
        print("train acc: {}".format(100.*correct/total))

        # evaluate performance on test set
        net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            # loop over all the batch
            for _, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                value, pred = outputs.max(1)
                total += targets.size(0)
                correct += pred.eq(targets).sum().item()

        # Save model if it is better than older one
        acc = 100.*correct/total
        if acc > best_acc:
            state = {
                'net': net.state_dict(),
                'epoch': epoch,
            }
            if not os.path.isdir('pretrained'):
                os.mkdir('pretrained')

            torch.save(state, out_path)
            best_acc = acc
        print("test acc: {}".format(acc))
        print("best test acc: {}".format(best_acc))

        scheduler.step()

# evaluate the model
net = models(args.model)

# load best model checkpoint
net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)

checkpoint = torch.load(out_path)

print("\nBest model was created in {} epoch:".format(checkpoint['epoch']))

net.load_state_dict(checkpoint['net'])
net.eval()

# classed list:
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# record all class corrects and frequency
class_correct = np.zeros((10, 10))
class_freq = np.zeros((10, 10))

correct = 0
with torch.no_grad():
    # get predict result for every batch
    for idxs, (inputs, labels) in enumerate(testloader):
        inputs, labels = inputs.to(device), labels.to(device)

        # get output of net
        out = net(inputs)

        # set the max probability class to 1, present as our prediction
        _, pred = out.max(1)

        Correct = pred[pred == labels]
        Wrong = pred[pred != labels]
        Truth = labels[pred != labels]

        for i in range(10):
            class_correct[i][i] += Correct.eq(i).sum().item()
            class_freq[i][:] += labels.eq(i).sum().item()

        for truth, wrong in zip(Truth, Wrong):
            class_correct[truth][wrong] += 1

        correct += pred.eq(labels).sum().item()

acc = 100.*correct/10000
print("Accuracy of {} on test-set: {}\n".format(args.model, acc))

class_correct /= class_freq

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(class_correct)

# Create colorbar
cbar = ax.figure.colorbar(im, ax=ax)

# show all ticks and label
ax.set_xlabel("Prediction")
ax.set_ylabel("Truth")
ax.set_xticks(np.arange(len(classes)))
ax.set_yticks(np.arange(len(classes)))
ax.set_xticklabels(classes, rotation=45)
ax.set_yticklabels(classes)

# loop over all the grid and show their value (set precision as 2)
for i in range(len(classes)):
    for j in range(len(classes)):
        text = ax.text(j, i, class_correct[i][j],
                       ha="center", va="center", color="w")

# set title
ax.set_title("Confusion Matrix")
fig.tight_layout()
plt.show()
