import torch
import torch.nn as nn
from torch.nn import Sequential,Conv2d,BatchNorm2d,MaxPool2d,Flatten,ReLU,Softmax,Linear


class LeNet(nn.Module):
    def __init__(self,input_size=32,output_size=10):
        super().__init__()
        self.Conv = Sequential(
            Conv2d(in_channels=3,out_channels=6,kernel_size=5,stride=1),
            MaxPool2d(kernel_size=(2,2)),
            BatchNorm2d(6),

            Conv2d(in_channels=6,out_channels=16,kernel_size=5,stride=1),
            MaxPool2d(kernel_size=(2,2)),
            BatchNorm2d(16),

            Flatten(),
        )
        self.fc = Sequential(
            Linear(in_features=400,out_features=120),
            ReLU(inplace=True),
            Linear(in_features=120,out_features=84),
            ReLU(inplace=True),
            Linear(in_features=84,out_features=output_size),
            Softmax(dim=1)
        )
        self.model = Sequential(
            self.Conv,
            self.fc
        )
    def forward(self,x):
        return self.model(x)
    
if __name__ == "__main__":
    net = LeNet()
    for name,param in net.named_parameters():
        print(name)