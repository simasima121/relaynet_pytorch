import torch
import numpy as np
from torch.nn.modules.loss import _Loss
from torch.autograd import Function, Variable
import torch.nn as nn
import torch.nn.functional as F


class DiceCoeff(nn.Module):
    """Dice coeff for individual examples"""
    def __init__(self):
        super(DiceCoeff, self).__init__()

    def forward(self, input, target):
        # self.save_for_backward(input, target)


        # print(target)

        self.inter = torch.dot(input, target) + 0.0001
        self.union = torch.sum(input**2) + torch.sum(target**2) + 0.0001

        t = 2*self.inter.float()/self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    # def backward(self, grad_output):
    #
    #     input, target = self.saved_variables
    #     grad_input = grad_target = None
    #
    #     if self.needs_input_grad[0]:
    #         grad_input = grad_output * 2 * (target * self.union + self.inter) \
    #                      / self.union * self.union
    #     if self.needs_input_grad[1]:
    #         grad_target = None
    #
    #     return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = Variable(torch.FloatTensor(1).cuda().zero_())
    else:
        s = Variable(torch.FloatTensor(1).zero_())

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i+1)


class DiceLoss(_Loss):
    def forward(self, input, target):
        return 1 - dice_coeff(input, target)


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average)

    def forward(self, inputs, targets):
        return self.nll_loss(inputs, targets)

class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.dice_loss=DiceLoss()
        self.cross_entropy_loss = CrossEntropyLoss2d()

    def forward(self, input, target, target_bin, weight):

        # TODO: why?
        target_bin = target_bin.type(torch.FloatTensor).cuda()
        target = target.type(torch.LongTensor).cuda()

        # print(type(input.data))
        # print(target.data.shape)
        # print(type(target_bin.data))
        # print(type(weight.data))

        y1 = self.dice_loss(input, target_bin)
        y2 = torch.mean(torch.mul(self.cross_entropy_loss.forward(input, target), weight))
        y = torch.add(y1,y2)
        # print(y1.data.shape)
        # print(y2.data.shape)

        return y