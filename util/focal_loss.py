'''
Code referenced from: 
https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
'''

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


__Author__ = 'Quanhao Guo'
__Date__ = '2021.05.03.16.30'


class FocalLoss(nn.Layer):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float, int)): self.alpha = paddle.to_tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = paddle.to_tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.reshape([input.shape[0], input.shape[1],-1]) # N,C,H,W => N,C,H*W
            input = input.transpose([0, 2, 1]) # N,C,H*W => N,H*W,C
            input = input.reshape([-1,input.shape[2]]) # N,H*W,C => N*H*W,C
        target = target.reshape([-1])

        logpt = F.log_softmax(input)
        logpt = paddle.nn.functional.one_hot(target, num_classes=2).multiply(logpt).sum(1)
        pt = logpt.exp()

        if self.alpha is not None:
            assert self.alpha.type == input.type
            at = self.alpha.unsqueeze(1).cast('float32').multiply(paddle.t(paddle.nn.functional.one_hot(target, num_classes=2))).sum(0)
            logpt = logpt * at

        loss = -1 * (1-pt)**self.gamma * logpt
        
        if self.size_average: 
            return loss.mean()
        else: 
            return loss.sum()
