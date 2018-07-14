import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable


class SGVLB(nn.Module):
    def __init__(self, net, train_size):
        super(SGVLB, self).__init__()
        self.train_size = train_size
        self.net = net

    def forward(self, input, target, kl_weight=1.0):
        assert not target.requires_grad
        kl = 0.0
        for module in self.net.modules():
            if hasattr(module, 'kl_reg'):
                kl = kl + module.kl_reg()
        return F.cross_entropy(input, target, size_average=True) * self.train_size + kl_weight * kl

    def get_kl(self):
        kl = 0.0
        for module in self.net.modules():
            if hasattr(module, 'kl_reg'):
                kl = kl + module.kl_reg()
        return kl


def lr_linear(epoch_num, decay_start, total_epochs, start_value):
    if epoch_num < decay_start:
        return start_value
    return start_value*float(total_epochs-epoch_num)/float(total_epochs-decay_start)


def logit2acc(outputs, targets):
    return np.mean(outputs.cpu().numpy().argmax(axis=1) == targets.data.cpu().numpy())


def kl_ard(log_alpha):
    return 0.5 * Variable.sum(Variable.log1p(Variable.exp(-log_alpha)))


def kl_loguni(log_alpha):
    k1, k2, k3 = 0.63576, 1.8732, 1.48695
    C = -k1
    mdkl = k1 * F.sigmoid(k2 + k3 * log_alpha) - 0.5 * Variable.log1p(Variable.exp(-log_alpha)) + C
    kl = -Variable.sum(mdkl)
    return kl
