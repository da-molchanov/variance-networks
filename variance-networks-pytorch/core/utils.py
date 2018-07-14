import torch
from torch.autograd import Variable
import torch.nn.functional as F
from core import metrics
import numpy as np


def where(cond, xt, xf):
    ret = torch.zeros_like(xt)
    ret[cond] = xt[cond]
    ret[cond ^ 1] = xf[cond ^ 1]
    return ret


def logmeanexp(x, dim=None, keepdim=False):
    """Stable computation of log(mean(exp(x))"""
    if dim is None:
        x, dim = x.view(-1), 0
    xm, _ = torch.max(x, dim, keepdim=True)
    x = xm + torch.log(torch.mean(torch.exp(x - xm), dim, keepdim=True))
    return x if keepdim else x.squeeze(dim)
    
    # It became broken: strange inf overflow error O_o
    # Let's hope all elements are finite
    # x = where((xm == float('inf')) | (xm == float('-inf')),
              # xm,
              # xm + torch.log(torch.mean(torch.exp(x - xm), dim, keepdim=True)))
    # return x if keepdim else x.squeeze(dim)


def adjust_learning_rate(optimizer, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def evaluate(net, dataloader, num_ens=1):
    """Calculate ensemble accuracy and NLL"""
    accs = []
    nlls = []
    for i, (inputs, labels) in enumerate(dataloader):
        inputs, labels = Variable(inputs.cuda(async=True)), Variable(labels.cuda(async=True))
        outputs = torch.zeros(inputs.shape[0], net.num_classes, num_ens).cuda()
        for j in range(num_ens):
            outputs[:, :, j] = F.log_softmax(net(inputs), dim=1).data
        accs.append(metrics.logit2acc(logmeanexp(outputs, dim=2), labels))
        nlls.append(F.nll_loss(Variable(logmeanexp(outputs, dim=2)), labels, size_average=False).data.cpu().numpy())
    return np.mean(accs), np.sum(nlls)
