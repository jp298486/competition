import torch.nn as nn
import torch
import torch.nn.functional as F
'''
Ref:
https://github.com/joe-siyuan-qiao/WeightStandardization#weight-standardization-on-computer-vision-tasks

目的:
    由於輸入的是單張 image ，意即 batch_size = 1 for testing 
    所以原先使用batch_normalize 的method 會使得預測結果變差
    故使用此一Convolution + group_normalize
'''
class WS_Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1,
                padding = 0, dilation = 1, groups = 1,  bias = True):
        super(WS_Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
            padding , dilation , groups,  bias)
    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim= 1, keepdim=True).mean(dim =2, 
                                            keepdim=True).mean(dim=3,keepdim=True)
        weight = weight - weight_mean
        
        std = weight.view(weight.size(0), -1).std(dim = 1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)

        return F.conv2d(x, weight, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)