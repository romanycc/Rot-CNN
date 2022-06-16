from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils import data
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import numpy as np
import warnings
from PIL import Image
import os
import os.path
import numpy as np
import torch
import codecs
import string
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.error import URLError
import shutil


test_losses = []
train_losses = []
    
def flip_filter(FN):
    ANS = torch.rot90(FN, 1, [2, 3])+ torch.rot90(FN, 2, [2, 3])+torch.rot90(FN, 3, [2, 3])+ FN
    x = torch.flip(FN,[3])
    ANS = ANS + torch.rot90(x, 1, [2, 3])+ torch.rot90(x, 2, [2, 3])+torch.rot90(x, 3, [2, 3])+ x
    ANS = ANS/8
    return ANS

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1) 
        self.conv2 = nn.Conv2d(32, 64, 5, 1) 
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(7744, 128) 
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output		

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    tmp=0
    for batch_idx, (data, target) in enumerate(train_loader):
        tmp=tmp+1
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        total_loss+= loss.item()
        loss.backward()     
        optimizer.step()   
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            total_loss = 0
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        tmp=0
        for data, target in test_loader:
            tmp=tmp+1
            data, target = data.to(device), target.to(device)        
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  
            pred = output.argmax(dim=1, keepdim=True) 
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nAverage loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 3)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    #define argument
    parser.add_argument('--train', action='store_true', default=False,
                        help='Train Model or Load Model')
    parser.add_argument('--augment', action='store_true', default=False,
                        help='Data Augmentation or not')
    parser.add_argument('--dataset', type=int, default=1,
                        help='1 for MNIST,2 for FashionMNIST')
    args = parser.parse_args()


    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    #transform
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    transform_rot=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.RandomAffine(degrees=(270,270))
        ])
    
    transform_rotran=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.RandomAffine(degrees=(0,360))
        ])
    
    #choose 1. MNIST or FashionMNIST  
    #       2. data augmentation or not
    if args.dataset==1:
        if args.augment==False:
            dataset1 = datasets.MNIST('../data', train=True, download=True,transform=transform)
        else:
            dataset1 = datasets.MNIST('../data', train=True, download=True,transform=transform_rotran)

        dataset2 = datasets.MNIST('../data', train=False,
                        transform=transform)
        dataset3 = datasets.MNIST('../data', train=False,
                        transform=transform_rot)
        dataset4 = datasets.MNIST('../data', train=False,
                        transform=transform_rotran)
    else:
        if args.augment==False:
            dataset1 = datasets.FashionMNIST('../data', train=True, download=True,transform=transform)
        else:
            dataset1 = datasets.FashionMNIST('../data', train=True, download=True,transform=transform_rotran)

        dataset2 = datasets.FashionMNIST('../data', train=False,
                        transform=transform)
        dataset3 = datasets.FashionMNIST('../data', train=False,
                        transform=transform_rot)
        dataset4 = datasets.FashionMNIST('../data', train=False,
                        transform=transform_rotran)
    
    
    
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    test_loader_rot = torch.utils.data.DataLoader(dataset3, **test_kwargs)
    test_loader_ranrot = torch.utils.data.DataLoader(dataset4, **test_kwargs)

    model = Net().to(device)

    #train or test
    if args.train==True:
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch)
            print("【 Train set 】")
            test(model, device, train_loader)
            print("【 Test set 】")
            test(model, device, test_loader)
            print("【 Test set rot 270 degree 】")
            test(model, device, test_loader_rot)
            print("【 Test set rot random degree 】")
            test(model, device, test_loader_ranrot)
            torch.save(model.state_dict(), './parameter/model.pt')
            scheduler.step()
    elif args.train==False:
        if args.augment==True and args.dataset==1:
            model.load_state_dict(torch.load('./parameter/CNN1_MNIST_AUG.pt'))
        if args.augment==True and args.dataset==2:
            model.load_state_dict(torch.load('./parameter/CNN1_FashionMNIST_AUG.pt'))
        if args.augment==False and args.dataset==1:
            model.load_state_dict(torch.load('./parameter/CNN1_MNIST.pt'))
        if args.augment==False and args.dataset==2:
            model.load_state_dict(torch.load('./parameter/CNN1_FashionMNIST.pt'))
        print("【 Train set 】")
        test(model, device, train_loader)
        print("【 Test set 】")
        test(model, device, test_loader)
        print("【 Test set rot 270 degree 】")
        test(model, device, test_loader_rot)
        print("【 Test set rot random degree 】")
        test(model, device, test_loader_ranrot)

if __name__ == '__main__':
    main()
