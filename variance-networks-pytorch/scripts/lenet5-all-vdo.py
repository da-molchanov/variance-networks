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
        self.conv1 = layers.ConvVDO(1, 20, 5, padding=0, alpha_shape=(1, 1, 1, 1))
        self.relu1 = nn.ReLU(True)
        self.pool1 = nn.MaxPool2d(2, padding=0)

        self.conv2 = layers.ConvVDO(20, 50, 5, padding=0, alpha_shape=(1, 1, 1, 1))
        self.relu2 = nn.ReLU(True)
        self.pool2 = nn.MaxPool2d(2, padding=0)

        self.flatten = layers.FlattenLayer(800)
        self.dense1 = layers.LinearVDO(800, 500)
        self.bn1 = nn.BatchNorm1d(500)
        self.relu3 = nn.ReLU()

        self.dense2 = layers.LinearVDO(500, 10)


fmt = {'tr_loss': '3.1e',
       'tr_acc': '.4f',
       'te_acc_det': '.4f',
       'te_acc_stoch': '.4f',
       'te_acc_ens': '.4f',
       'te_acc_perm_sigma': '.4f',
       'te_acc_zero_mean': '.4f',
       'te_acc_perm_sigma_ens': '.4f',
       'te_acc_zero_mean_ens': '.4f',
       'te_nll_det': '.4f',
       'te_nll_stoch': '.4f',
       'te_nll_ens': '.4f',
       'te_nll_perm_sigma': '.4f',
       'te_nll_zero_mean': '.4f',
       'te_nll_perm_sigma_ens': '.4f',
       'te_nll_zero_mean_ens': '.4f',
       'time': '.3f'}
fmt = {**fmt, **{'la%d' % i: '.4f' for i in range(4)}}
logger = Logger("lenet5-VDO", fmt=fmt)

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

    # Zero-mean
    net.train()
    net.dense1.set_flag('zero_mean', True)
    acc, nll = utils.evaluate(net, testloader, num_ens=1)
    net.dense1.set_flag('zero_mean', False)
    logger.add(epoch, te_nll_zero_mean=nll, te_acc_zero_mean=acc)

    # Permuted sigmas
    net.train()
    net.dense1.set_flag('permute_sigma', True)
    acc, nll = utils.evaluate(net, testloader, num_ens=1)
    net.dense1.set_flag('permute_sigma', False)
    logger.add(epoch, te_nll_perm_sigma=nll, te_acc_perm_sigma=acc)

    # Zero-mean test-time averaging
    net.train()
    net.dense1.set_flag('zero_mean', True)
    acc, nll = utils.evaluate(net, testloader, num_ens=20)
    net.dense1.set_flag('zero_mean', False)
    logger.add(epoch, te_nll_zero_mean_ens=nll, te_acc_zero_mean_ens=acc)

    # Permuted sigmas test-time averaging
    net.train()
    net.dense1.set_flag('permute_sigma', True)
    acc, nll = utils.evaluate(net, testloader, num_ens=20)
    net.dense1.set_flag('permute_sigma', False)
    logger.add(epoch, te_nll_perm_sigma_ens=nll, te_acc_perm_sigma_ens=acc)

    logger.add(epoch, time=time()-t0)
    las = [np.mean(net.conv1.log_alpha.data.cpu().numpy()),
           np.mean(net.conv2.log_alpha.data.cpu().numpy()),
           np.mean(net.dense1.log_alpha.data.cpu().numpy()),
           np.mean(net.dense2.log_alpha.data.cpu().numpy())]

    logger.add(epoch, **{'la%d' % i: las[i] for i in range(4)})
    logger.iter_info()
    logger.save(silent=True)
    torch.save(net.state_dict(), logger.checkpoint)

logger.save()
