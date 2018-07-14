import torch
from torch.autograd import Variable
import torch.nn as nn
from core import layers
import torch.optim as optim
from core import metrics
from core import utils
from time import time
import numpy as np
from core.logger import Logger

import torchvision
import torchvision.transforms as transforms


class LeNet5(layers.ModuleWrapper):

    def __init__(self):
        super(LeNet5, self).__init__()
        self.num_classes = 10
        self.conv1 = nn.Conv2d(1, 20, 5, padding=0)
        self.relu1 = nn.Tanh()
        self.pool1 = nn.MaxPool2d(2, padding=0)

        self.conv2 = nn.Conv2d(20, 50, 5, padding=0)
        self.relu2 = nn.Tanh()
        self.pool2 = nn.MaxPool2d(2, padding=0)

        self.flatten = layers.FlattenLayer(800)
        self.do1 = nn.Dropout(0.5)
        self.dense1 = nn.Linear(800, 500)
        self.relu3 = nn.Tanh()

        self.dense2 = nn.Linear(500, 10)


fmt = {'tr_loss': '3.1e',
       'tr_acc': '.4f',
       'te_acc_det': '.4f',
       'te_acc_stoch': '.4f',
       'te_acc_ens': '.4f',
       'te_nll_det': '.4f',
       'te_nll_stoch': '.4f',
       'te_nll_ens': '.4f',
       'time': '.3f'}
logger = Logger("lenet5-DO", fmt=fmt)

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

criterion = metrics.SGVLB(net, 60000.).cuda()
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
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        accs.append(metrics.logit2acc(outputs.data, labels))  # probably a bad way to calculate accuracy
        training_loss += loss.cpu().data.numpy()[0]

    logger.add(epoch, tr_loss=training_loss/steps, tr_acc=np.mean(accs))

    # Deterministic test
    net.eval()
    acc, nll = utils.evaluate(net, testloader, num_ens=1)
    logger.add(epoch, te_nll_det=nll, te_acc_det=acc)

    # Stochastic test
    net.train()
    acc, nll = utils.evaluate(net, testloader, num_ens=1)
    logger.add(epoch, te_nll_stoch=nll, te_acc_stoch=acc)

    # Test-time averaging
    net.train()
    acc, nll = utils.evaluate(net, testloader, num_ens=20)
    logger.add(epoch, te_nll_ens=nll, te_acc_ens=acc)

    logger.add(epoch, time=time()-t0)
    logger.iter_info()
    logger.save(silent=True)
    torch.save(net.state_dict(), logger.checkpoint)

logger.save()
