#!/usr/bin/env python
"""
Train a H-DNN on MNIST dataset.
Author: Clara Galimberti (clara.galimberti@epfl.ch)
Usage:
python run_MNIST.py          --net_type      [MODEL NAME]            \
                             --n_layers      [NUMBER OF LAYERS]      \
                             --gpu           [GPU ID]
Flags:
  --net_type: Network model to use. Available options are: MS1, H1_J1, H1_J2.
  --n_layers: Number of layers for the chosen the model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np

from integrators.integrators import MS1, H1
from regularization.regularization import regularization
import argparse


class Net(nn.Module):
    def __init__(self, nf=8, n_layers=4, h=0.5, net_type='H1_J1'):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=nf, kernel_size=3, stride=1, padding=1)
        if net_type == 'MS1':
            self.hamiltonian = MS1(n_layers=n_layers, t_end=h * n_layers, nf=nf)
        elif net_type == 'H1_J1':
            self.hamiltonian = H1(n_layers=n_layers, t_end=h * n_layers, nf=nf, select_j='J1')
        elif net_type == 'H1_J2':
            self.hamiltonian = H1(n_layers=n_layers, t_end=h * n_layers, nf=nf, select_j='J2')
        else:
            raise ValueError("%s model is not yet implemented for MNIST" % net_type)
        self.hamiltonian = MS1(n_layers=n_layers, t_end=h * n_layers, nf=nf)
        self.fc_end = nn.Linear(nf*28*28, 10)
        self.nf = nf

    def forward(self, x):
        x = self.conv1(x)
        x = self.hamiltonian(x)
        x = x.reshape(-1, self.nf*28*28)
        x = self.fc_end(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(model, device, train_loader, optimizer, epoch, alpha, out):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        K = model.hamiltonian.getK()
        b = model.hamiltonian.getb()
        for j in range(int(model.hamiltonian.n_layers) - 1):
            loss = loss + regularization(alpha, h, K, b)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0 and out>0:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            print('\tTrain Epoch: {:2d} [{:5d}/{} ({:2.0f}%)]\tLoss: {:.6f}\tAccuracy: {}/{}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), correct, len(data)))


def test(model, device, test_loader, out):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    if out > 0:
        print('Test set:\tAverage loss: {:.4f}, Accuracy: {:5d}/{} ({:.2f}%)'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    return correct


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--net_type', type=str, default='H1_J1')
    parser.add_argument('--n_layers', type=int, default=1)
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()  # not no_cuda and
    batch_size = 100
    test_batch_size = 1000
    lr = 0.04
    gamma = 0.8
    epochs = 2
    seed = np.random.randint(0, 1000)
    torch.manual_seed(seed)
    np.random.seed(seed)

    out = 1

    if args.net_type == 'MS1':
        h = 0.4
        wd = 1e-3
        alpha = 1e-3
    elif args.net_type == 'H1_J1':
        h = 0.5
        wd = 4e-3
        alpha = 8e-3
    elif args.net_type == 'H1_J2':
        h = 0.05
        wd = 2e-4
        alpha = 1e-3
    else:
        raise ValueError("%s model is not yet implemented" % args.net_type)

    # Define the net model
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 20, 'pin_memory': True} if use_cuda else {}
    model = Net(nf=8, n_layers=args.n_layers, h=h, net_type=args.net_type).to(device)

    print("\n------------------------------------------------------------------")
    print("MNIST dataset - %s-DNN - %i layers" % (args.net_type, args.n_layers))
    print("== sgd with Adam (lr=%.1e, weight_decay=%.1e, gamma=%.1f, max_epochs=%i, alpha=%.1e, minibatch=%i)" %
          (lr, wd, gamma, epochs, alpha, batch_size))

    best_acc = 0
    best_acc_train = 0

    # Load train data
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    # Load test data
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=test_batch_size, shuffle=True, **kwargs)

    # Define optimization algorithm
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # Scheduler for learning_rate parameter
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    # Training
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch, alpha, out)
        test_acc = test(model, device, test_loader, out)
        # Results over training set after training
        train_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                train_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
        train_loss /= len(train_loader.dataset)
        if out > 0:
            print('Train set:\tAverage loss: {:.4f}, Accuracy: {:5d}/{} ({:.2f}%)'.format(
                train_loss, correct, len(train_loader.dataset),
                100. * correct / len(train_loader.dataset)))
        scheduler.step()
        if best_acc < test_acc:
            best_acc = test_acc
            best_acc_train = correct

    print("\nNetwork trained!")
    print('Test accuracy: {:.2f}%  - Train accuracy: {:.3f}% '.format(
         100. * best_acc / len(test_loader.dataset), 100. * best_acc_train / len(train_loader.dataset)))
    print("------------------------------------------------------------------\n")

