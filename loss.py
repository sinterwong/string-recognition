import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import torch.nn.functional as F


# AMSoftmax 层的pytorch实现，两个重要参数 scale，margin（不同难度和量级的数据对应不同的最优参数）
class AMSoftmax(nn.Module):
    def __init__(self, m=0.3, s=15):
        super(AMSoftmax, self).__init__()
        self.m = m
        self.s = s
        self.ce = nn.CrossEntropyLoss()

    def forward(self, x, lb):
        costh = x
        lb_view = lb.view(-1, 1)
        if lb_view.is_cuda: lb_view = lb_view.cpu()
        delt_costh = torch.zeros(costh.size()).scatter_(1, lb_view, self.m)
        if x.is_cuda: delt_costh = delt_costh.cuda()
        costh_m = costh - delt_costh
        costh_m_s = self.s * costh_m
        loss = self.ce(costh_m_s, lb)
        return loss
