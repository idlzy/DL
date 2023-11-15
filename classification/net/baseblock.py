import torch
import torch.nn as nn
from torch.nn import Sequential,Conv2d,BatchNorm2d,MaxPool2d,Flatten,ReLU,Softmax,Linear,Dropout


class ConvLayer(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=-1,downsample=True):
        super().__init__()
        self.downsample = downsample
        if padding==-1:
            padding = (kernel_size-1)//2
        self.Conv = Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding)
        self.Activation = ReLU(inplace=True)
        self.Maxpool = MaxPool2d(kernel_size=(2,2))
        self.BatchNorm = BatchNorm2d(num_features=out_channels)
    def forward(self,x):
        x = self.Conv(x)
        x = self.Activation(x)
        if self.downsample:
            x = self.Maxpool(x)
        x = self.BatchNorm(x)
        return x
    
class FullConnectLayer(nn.Module):
    def __init__(self,in_features,out_features,drop_p=0,softmax=False):
        super().__init__()
        self.drop_p = drop_p
        self.fc = Linear(in_features=in_features,out_features=out_features)
        self.drop = Dropout(drop_p)
        self.Activate = Softmax(dim=1) if softmax else ReLU(inplace=True)
    def forward(self,x):
        x = self.fc(x)
        if self.drop_p>0:
            x = self.drop(x)
        x = self.Activate(x)
        return x
    

