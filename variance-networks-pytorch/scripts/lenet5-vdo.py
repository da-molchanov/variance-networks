import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import functional as F
from core import layers
import torch.optim as optim
from core import metrics
from core import utils
from time import time
import numpy as np
from core.logger import Logger

import torchvision
import torchvision.transforms as transforms

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--vdo1", action='store_true', default=False)
parser.add_argument("--vdo2", action='store_true', default=False)
parser.add_argument("--vdo3", action='store_true', default=False)
parser.add_argument("--vdo4", action='store_true', default=False)
parser.add_argument("--train-ens", type=int, default=1)
parser.add_argument("--no-biases", action='store_true', default=False)
parser.add_argument("--tanh", action='store_true', default=False)
flags = parser.parse_args()


class LeNet5(layers.ModuleWrapper):

    def __init__(self):
        super(LeNet5, self).__init__()
        self.num_classes = 10
        if flags.tanh:
            nonlinearity = nn.Tanh
        else:
            nonlinearity = nn.ReLU

        # Conv-BN-Tanh-Pool
        if flags.vdo1:
            self.conv1 = layers.ConvVDO(1, 20, 5, alpha_shape=(1,1), padding=0, bias=False)
        else:
            self.conv1 = nn.Conv2d(1, 20, 5, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(20, affine=not flags.no_biases)
        self.relu1 = nonlinearity()
        self.pool1 = nn.MaxPool2d(2, padding=0)

        # Conv-BN-Tanh-Pool
        if flags.vdo2:
            self.conv2 = layers.ConvVDO(20, 50, 5, alpha_shape=(1,1), padding=0, bias=False)
        else:
            self.conv2 = nn.Conv2d(20, 50, 5, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(50, affine=not flags.no_biases)
        self.relu2 = nonlinearity()
        self.pool2 = nn.MaxPool2d(2, padding=0)

        self.flatten = layers.FlattenLayer(800)

        if not (flags.vdo1 or flags.vdo2 or flags.vdo3 or flags.vdo4):
            self.do1 = nn.Dropout(0.5)

        # Dense-BN-Tanh
        if flags.vdo3:
            self.dense1 = layers.LinearVDO(800, 500, alpha_shape=(1,1), bias=False)
        else:
            self.dense1 = nn.Linear(800, 500, bias=False)
        self.bn3 = nn.BatchNorm1d(500, affine=not flags.no_biases)
        self.relu3 = nonlinearity()

        # Dense
        if flags.vdo4:
            self.dense2 = layers.LinearVDO(500, 10, alpha_shape=(1,1), bias=False)
        else:
            self.dense2 = nn.Linear(500, 10, bias=False)


fmt = {'kl': '3.3e',
       'tr_loss': '3.3e',
       'tr_acc': '.4f',
       'zero_mean_tr_loss': '3.3e',
       'zero_mean_tr_acc': '.4f',
       'te_acc_ens100': '.4f',
       'te_acc_stoch': '.4f',
       'te_acc_ens10': '.4f',
       'te_acc_perm_sigma': '.4f',
       'te_acc_zero_mean': '.4f',
       'te_acc_perm_sigma_ens': '.4f',
       'te_acc_zero_mean_ens': '.4f',
       'te_nll_ens100': '.4f',
       'te_nll_stoch': '.4f',
       'te_nll_ens10': '.4f',
       'te_nll_perm_sigma': '.4f',
       'te_nll_zero_mean': '.4f',
       'te_nll_perm_sigma_ens': '.4f',
       'te_nll_zero_mean_ens': '.4f',
       'time': '.3f'}
logger = Logger("lenet5-VDO-layerwise", fmt=fmt)

net = LeNet5()
net.cuda()
logger.print(net)

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                      download=True, transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=200,
                                          shuffle=True, num_workers=4, pin_memory=True)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                     download=True, transform=transforms.ToTensor())
testloader = torch.utils.data.DataLoader(testset, batch_size=200,
                                         shuffle=False, num_workers=4, pin_memory=True)

criterion = metrics.SGVLB(net, len(trainset)).cuda()
optimizer = optim.Adam(net.parameters(), lr=0.001)

epochs = 200
lr_start = 1e-3
for epoch in range(epochs):  # loop over the dataset multiple times
    t0 = time()
    utils.adjust_learning_rate(optimizer, metrics.lr_linear(epoch, 0, epochs, lr_start))
    net.train()
    training_loss = 0
    accs = []
    steps = 0
    for i, (inputs, labels) in enumerate(trainloader, 0):
        steps += 1
        inputs, labels = Variable(inputs.cuda(async=True)), Variable(labels.cuda(async=True))

        optimizer.zero_grad()

        outputs = Variable(torch.zeros(inputs.shape[0], net.num_classes, flags.train_ens).cuda())
        for j in range(flags.train_ens):
            outputs[:, :, j] = F.log_softmax(net(inputs), dim=1)
        log_outputs = utils.logmeanexp(outputs, dim=2)

        loss = criterion(log_outputs, labels)
        loss.backward()
        optimizer.step()

        accs.append(metrics.logit2acc(log_outputs.data, labels))
        training_loss += loss.cpu().data.numpy()[0]

    # ELBO evaluation
    net.train()
    training_loss = 0
    steps = 0
    accs = []
    for i, (inputs, labels) in enumerate(trainloader, 0):
        steps += 1
        inputs, labels = Variable(inputs.cuda(async=True)), Variable(labels.cuda(async=True))
        for j in range(10):
            outputs = net(inputs).detach()
            training_loss += criterion(outputs, labels).cpu().data.numpy()[0]/10.0
        accs.append(metrics.logit2acc(outputs.data, labels))
    logger.add(epoch, kl=criterion.get_kl(), tr_loss=training_loss/steps, tr_acc=np.mean(accs))

    # zero-mean ELBO evaluation
    net.train()
    net.set_flag('zero_mean', True)
    training_loss = 0
    steps = 0
    accs = []
    for i, (inputs, labels) in enumerate(trainloader, 0):
        steps += 1
        inputs, labels = Variable(inputs.cuda(async=True)), Variable(labels.cuda(async=True))
        for j in range(10):
            outputs = net(inputs).detach()
            training_loss += criterion(net(inputs), labels).cpu().data.numpy()[0] / 10.0
        accs.append(metrics.logit2acc(outputs.data, labels))
    logger.add(epoch, zero_mean_tr_loss=training_loss / steps, zero_mean_tr_acc=np.mean(accs))
    net.set_flag('zero_mean', False)

    # Stochastic test
    net.train()
    acc, nll = utils.evaluate(net, testloader, num_ens=1)
    logger.add(epoch, te_nll_stoch=nll, te_acc_stoch=acc)

    # Test-time averaging
    net.train()
    acc, nll = utils.evaluate(net, testloader, num_ens=10)
    logger.add(epoch, te_nll_ens10=nll, te_acc_ens10=acc)

    logger.add(epoch, time=time()-t0)
    logger.iter_info()
    logger.save(silent=True)
    torch.save(net.state_dict(), logger.checkpoint)

logger.save()
