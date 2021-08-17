import torch.nn as nn
import torch
import torch.nn.functional

from .Weight_Stand_Conv import WS_Conv2d

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


# if __name__ == '__main__' :
#     x = torch.rand(3, 64, 16, 16)
#     model_1 = Base_block_GN(64, 128, stride=1)
#     model_1_1 = Individual_block_GN(64, 128, stride=1)
#     model_2 = Bottleneck_block_GN(64, 32, stride=2)
    
#     print('base_block out_shape= ', model_1(x).size())
#     print('Individual_block_GN out_shape= ', model_1_1(x).size())
#     print('bottle_neck_block out_shape = ',model_2(x).size())
    
IsBottleNeck_list = [32, 64, 128, 256]
IsBase_list = [64, 128, 256, 512]
