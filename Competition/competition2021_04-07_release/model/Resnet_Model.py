
import sys
import os 

import torch.nn as nn
import torch
import torch.nn.functional

from model.Weight_Stand_Conv import WS_Conv2d

'''
base block use WS_Conv + GroupNorm
'''
# base block (3x3, 3x3) == out_channel == > planes, planes
class Base_block_GN(nn.Module):
    def __init__(self, in_planes, planes, stride = 1, GN_num = 32):
        super(Base_block_GN, self).__init__()

        self.stride = stride
        self.conv = nn.Sequential(
            WS_Conv2d(in_planes, planes, kernel_size=(3, 3), stride = self.stride, padding = 1),
            nn.GroupNorm(GN_num, planes),
            nn.ReLU(),
            WS_Conv2d(planes, planes, kernel_size=(3, 3), stride = 1, padding=1),
            nn.GroupNorm(GN_num, planes)
        )

        self.shortcut = nn.Sequential()
        if self.stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                WS_Conv2d(in_planes, planes, kernel_size=1, stride=self.stride),
                nn.GroupNorm(GN_num, planes)
            )
        self.activate_out = nn.ReLU()
    def forward(self, x):
        out = self.conv(x)
        out += self.shortcut(x)
        out = self.activate_out(out)

        return out

# bottle_neck block(1x1, 3x3, 1x1) == out_channel == > planes, planes, planes*self.expansion
class Bottleneck_block_GN(nn.Module):
    def __init__(self, in_planes, planes, stride = 1, GN_num = 32, expansion = 4):
        super(Bottleneck_block_GN, self).__init__()
        self.stride = stride
        self.expansion = expansion

        self.conv = nn.Sequential(
            WS_Conv2d(in_planes, planes, kernel_size=(1, 1), bias=False),
            nn.GroupNorm(GN_num, planes),
            nn.ReLU(),
            WS_Conv2d(planes, planes, kernel_size=(3, 3), stride = self.stride, padding = 1, bias=False),
            nn.GroupNorm(GN_num, planes),
            nn.ReLU(),
            WS_Conv2d(planes, planes*self.expansion ,kernel_size=(1, 1), bias=False),
            nn.GroupNorm(GN_num, planes*self.expansion),
        )
        self.shortcut = nn.Sequential()
        if self.stride != 1 or in_planes != planes*self.expansion:
            self.shortcut = nn.Sequential(
                WS_Conv2d(in_planes, planes * self.expansion, kernel_size=(1, 1), stride = self.stride),
                nn.GroupNorm(GN_num, planes * self.expansion),
            )
        self.activate_out = nn.ReLU()
    def forward(self, x):
        out = self.conv(x)
        out += self.shortcut(x)
        out = self.activate_out(out)

        return out

# Individual Residual 
# Ref: https://ai-scholar.tech/zh/articles/materials-informatics/IRNet-DL
class Individual_block_GN(nn.Module):
    def __init__(self, in_planes, planes, stride = 1, GN_num = 32):
        super(Individual_block_GN, self).__init__()
        self.stride = stride

        self.conv1 = nn.Sequential(
            WS_Conv2d(in_planes, planes, kernel_size=(3, 3), stride = self.stride, padding = 1),
            nn.GroupNorm(GN_num, planes),
        )
        
        self.conv2 = nn.Sequential(
            WS_Conv2d(planes, planes, kernel_size=(3, 3), stride = 1, padding = 1),
            nn.GroupNorm(GN_num, planes),
        )
        self.shortcut1 = nn.Sequential()
        if self.stride != 1 or in_planes != planes:
            self.shortcut1 = nn.Sequential(
                WS_Conv2d(in_planes, planes, kernel_size=(1, 1), stride=self.stride),
                nn.GroupNorm(GN_num, planes)
            )
        self.shortcut2 = nn.Sequential(
            WS_Conv2d(planes, planes, kernel_size=(1, 1), stride=1),
            nn.GroupNorm(GN_num, planes)
        )
        self.activate_out1 = nn.ReLU()
        self.activate_out2 = nn.ReLU()
    def forward(self, x):
        out = self.conv1(x)
        out += self.shortcut1(x)
        y = self.activate_out1(out)
        out = self.conv2(y)
        out += self.shortcut2(y)
        out = self.activate_out2(out)

        return out


# model
class ResNet(nn.Module):
    def __init__(self, block, n_block=[2,2,2,2], in_planes = 64, n_class = 801, n_gn = 32, expansion = 1):
        super(ResNet, self).__init__()
        self.planes = in_planes
        self.expansion = expansion

        self.conv = nn.Sequential(
            WS_Conv2d(3, in_planes, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(n_gn, in_planes),
            nn.ReLU()
        )
        self.layer1 = self._make_layer(block, in_planes, n_block[0], stride=1, GN_num=n_gn)
        self.layer2 = self._make_layer(block, in_planes*2, n_block[1], stride=2, GN_num=n_gn)
        self.layer3 = self._make_layer(block, in_planes*4, n_block[2], stride=2, GN_num=n_gn)
        self.layer4 = self._make_layer(block, in_planes*8, n_block[3], stride=2, GN_num=n_gn)

        self.fc = nn.Linear(in_planes*8*self.expansion, n_class)
        self.activate = nn.LogSoftmax(dim = -1)
    
    def _make_layer(self, block, out_planes, n_blocks, stride, GN_num):
        strides = [stride] + [1] *(n_blocks - 1)
        layers = []
        for stride in strides :
            layers.append(block(self.planes, out_planes, stride, GN_num))
            self.planes = out_planes * self.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv(x)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = torch.nn.functional.avg_pool2d(out, out.size(2))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.activate(out)
        return out
# if __name__ == '__main__' :

# x = torch.randn(4, 3, 64, 64)
# model = ResNet(Bottleneck_block_GN, in_planes = 32, expansion = 4) # 差異 expansion
# model = ResNet(Base_block_GN, in_planes = 32, expansion = 1) 
