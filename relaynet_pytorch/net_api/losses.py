import torch
import numpy as np
from torch.nn.modules.loss import _Loss
from torch.autograd import Function, Variable
import torch.nn as nn
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
        inter = torch.dot(input, target) + 0.0001
        union = torch.sum(input ** 2) + torch.sum(target ** 2) + 0.0001

        t = 2 * inter.float() / union.float()
        return t


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = Variable(torch.FloatTensor(1).cuda().zero_())
    else:
        s = Variable(torch.FloatTensor(1).zero_())

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)

class DiceLoss(_Loss):
    def forward(self, output, target, weights=None, ignore_index=None):
        """
            output : NxCxHxW Variable
            target :  NxHxW LongTensor
            weights : C FloatTensor
            ignore_index : int index to ignore from loss
        """
        eps = 0.0001

        output = output.exp() # output is a torch.Tensor of shape torch.Size([1, 7, 512, 64])
        #print('Unique values in output var', np.unique(output.detach()))
        #print('Output shape', output.shape)

        encoded_target = output.detach() * 0 # encoded target is FloatTensor with shape torch.Size([1, 7, 512, 64])
        #print('Encoded Target: unique values {}, shape: {}'.format(np.unique(encoded_target), encoded_target.shape))
        #print('Target: unique values {}, shape: {}, Unsqueeze Shape {}'.format(np.unique(target), target.shape, new_target.shape))
        
        
        # Target is a torch.cuda.LongTensor of shape torch.Size([1, 1, 512, 64]) after unsqueeze
        
        if ignore_index is not None:
            mask = target == ignore_index
            target = target.clone()
            target[mask] = 0
            encoded_target.scatter_(1, target.unsqueeze(1), 1)
            mask = mask.unsqueeze(1).expand_as(encoded_target)
            encoded_target[mask] = 0
        else:
            encoded_target.scatter_(1, target.unsqueeze(1), 1)

        if weights is None: # weight is an integer of value 1
            weights = 1
              
        intersection = output * encoded_target # <class 'torch.Tensor'> with size torch.Size([1, 7, 512, 64])
        numerator = 2 * intersection.sum(0).sum(1).sum(1) # numerator is a torch.Tensor with torch.Size([7])
        denominator = output + encoded_target # denominator is a torch.Tensor with torch.Size([7])
        
        #print(type(2*intersection.sum(0).sum(1).sum(1)))
        
        #for i in range(intersection.size(0)):
        #    for j in range(intersection.size(1)):
        #        for k in range(intersection.size(2)):
        #            print(k)
        #            print(intersection[i][j][k].data)
                    
        if ignore_index is not None:
            denominator[mask] = 0
        denominator = denominator.sum(0).sum(1).sum(1) + eps
        loss_per_channel = weights * (1 - (numerator / denominator)) # loss_p_channel is a torch.Tensor

        return loss_per_channel / output.size(1) # output.size(1) is an integer of 7

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.CrossEntropyLoss(weight, size_average)

    def forward(self, inputs, targets):
        return self.nll_loss(inputs, targets)


class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.cross_entropy_loss = CrossEntropyLoss2d()
        self.dice_loss = DiceLoss()

    def forward(self, input, target, weight):
        # TODO: why?
        target = target.type(torch.LongTensor).cuda()
        input_soft = F.softmax(input,dim=1)
        #print("softmax returned values", np.unique(input_soft.detach()))
        #print("Softmax output Shape", input_soft.shape)
        #print('Target Shape',target.shape)

        y2 = torch.mean(self.dice_loss(input_soft, target))
        y1 = torch.mean(torch.mul(self.cross_entropy_loss.forward(input, target), weight))
        y = y1 + y2
        return y

