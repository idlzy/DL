import torch
import torch.nn as nn
from torch.nn import Sequential, Conv2d,  MaxPool2d, Linear, Dropout
from torch.nn import LocalResponseNorm, Flatten, Sigmoid, ReLU, Softmax, BatchNorm2d
from torch.nn import AvgPool2d,AdaptiveAvgPool2d
class BasicConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=-1,bias=True):
        super().__init__()
        if padding==-1:
            padding = (kernel_size-1)//2
        self.Conv2d = Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding,bias=bias)
        self.Activation = ReLU(inplace=True)
    def forward(self,x):
        x = self.Conv2d(x)
        x = self.Activation(x)
        return x


class ConvLayer(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=-1,downsample=True,bias=False):
        super().__init__()
        self.downsample = downsample
        if padding==-1:
            padding = (kernel_size-1)//2
        self.Conv = Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding,bias=bias)
        self.Activation = ReLU(inplace=True)
        self.Maxpool = MaxPool2d(kernel_size=(2,2))
        self.BatchNorm = BatchNorm2d(num_features=out_channels)
    def forward(self,x):
        x = self.Conv(x)
        x = self.BatchNorm(x)
        x = self.Activation(x)
        if self.downsample:
            x = self.Maxpool(x)
        return x
    
class FullConnectLayer(nn.Module):
    def __init__(self,in_features,out_features,drop_p=0,output_layer=False):
        super().__init__()
        self.drop_p = drop_p
        self.fc = Linear(in_features=in_features,out_features=out_features)
        self.drop = Dropout(drop_p)
        self.Activate = Softmax(dim=1) if output_layer else ReLU()
    def forward(self,x):
        x = self.fc(x)
        x = self.Activate(x)
        if self.drop_p>0:
            x = self.drop(x)
        return x
    

