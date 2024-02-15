import torch
from torch import nn
import torch.nn.functional as F

torch.cuda.empty_cache()

class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias

        self.conv1 = nn.Conv2d(self.in_channels,self.in_channels,self.kernel_size,self.stride,self.padding,self.dilation,groups=self.in_channels,bias=self.bias)
        self.bn = nn.BatchNorm2d(self.in_channels),
        self.relu = nn.ReLU()
        self.pointwise = nn.Conv2d(self.in_channels,self.out_channels,1,1,0,1,1,bias=bias)

    def forward(self,x):
        x = self.conv1(x)
        #x = self.bn(x)
        #x = self.relu(x)
        x = self.pointwise(x)
        return x
  
class Block(nn.Module):
    def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True):
        super(Block, self).__init__()

        self.in_filters = in_filters
        self.out_filters = out_filters
        self.reps = reps
        self.strides = strides
        self.start_with_relu = start_with_relu
        self.grow_first = grow_first

        self.skipbn = nn.BatchNorm2d(self.out_filters)
        if self.out_filters != self.in_filters or self.strides!=1:
            self.skip = nn.Conv2d(self.in_filters,self.out_filters,1,stride=self.strides, bias=False)

        else:
            self.skip=None

        self.relu = nn.ReLU(inplace=True)
        rep=[]

        filters=in_filters
        if grow_first:
          rep.append(self.relu)
          rep.append(SeparableConv2d(self.in_filters,self.out_filters,3,stride=self.strides,padding=1,bias=False)) # stride 수정
          rep.append(nn.BatchNorm2d(self.out_filters))
          filters = self.out_filters

        for i in range(reps-1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters,filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(self.in_filters,self.out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(self.out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        self.rep = nn.Sequential(*rep)

    def forward(self,inp):
        x = self.rep(inp)
        #print(self.skip is not None)
        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x+=skip
        return x
    
class ModifiedXception(nn.Module):
    def __init__(self):
        super(ModifiedXception, self).__init__()

        # Entry flow
        self.conv1 = nn.Conv2d(3, 32, 3,2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32,64,3,bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

        self.block1=Block(64,128,2,2,start_with_relu=False,grow_first=True)
        self.block2=Block(128,256,2,2,start_with_relu=True,grow_first=True)
        self.block3=Block(256,728,2,2,start_with_relu=True,grow_first=True)

        # Middle flow
        self.middle_flow = nn.Sequential(*[Block(728,728,3,1,start_with_relu=True,grow_first=False) for _ in range(16)])

        # Exit flow
        self.block8=Block(728,1024,2,2,start_with_relu=True,grow_first=True)
        self.conv3 = SeparableConv2d(1024,1536,3,1,1)
        self.bn3 = nn.BatchNorm2d(1536)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        # Entry flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.block1(x)
        low_level_feat = x
        x = self.block2(x)
        x = self.block3(x)

        # Middle flow
        x = self.middle_flow(x)

        # Exit flow
        x = self.block8(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        return low_level_feat, x