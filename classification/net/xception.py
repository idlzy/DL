from net.baseblock import *

class SeparableConv(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1):
        super().__init__()
        self.Conv3x3 = Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=3,stride=stride,padding=1,bias=False,groups=in_channels)
        self.Conv1x1 = Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,bias=False)
        self.BN = BatchNorm2d(num_features=out_channels)

    def forward(self,x):
        x = self.Conv3x3(x)
        x = self.Conv1x1(x)
        x = self.BN(x)
        return x
    
class SeparableConvBlocks(nn.Module):
    def __init__(self,in_channels,out_channels,mid_channels=-1):
        super().__init__()
        if mid_channels == -1:
            mid_channels = out_channels
        self.ConvBlocks = Sequential(
            SeparableConv(in_channels=in_channels,out_channels=mid_channels),
            SeparableConv(in_channels=mid_channels,out_channels=out_channels),
            MaxPool2d(kernel_size=(3,3),stride=2,padding=1)
        )
        self.Conv1x1Block = BasicConv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,stride=2,bias=False)
        self.BN = BatchNorm2d(num_features=out_channels)
    def forward(self,x):
        x1 = self.Conv1x1Block(x)
        x1 = self.BN(x1)
        x2 = self.ConvBlocks(x)
        x = x1+x2
        return x

class MiddleFlowBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.Blocks = Sequential(
            SeparableConv(in_channels=728,out_channels=728),
            SeparableConv(in_channels=728,out_channels=728),
            SeparableConv(in_channels=728,out_channels=728)
        )
    def forward(self,x):
        x1 = self.Blocks(x)
        x = x1+x
        return x
class Xception(nn.Module):
    def __init__(self,output_size=1000):
        super().__init__()
        self.EntryFlow = Sequential(
            BasicConv2d(in_channels=3,out_channels=32,kernel_size=3,stride=2,bias=False),
            BatchNorm2d(num_features=32),

            BasicConv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,bias=False),
            BatchNorm2d(num_features=64),

            SeparableConvBlocks(in_channels=64,out_channels=128),
            SeparableConvBlocks(in_channels=128,out_channels=256),
            SeparableConvBlocks(in_channels=256,out_channels=728),
        )
        self.MiddleFlow = []
        for i in range(8):
            self.MiddleFlow.append(MiddleFlowBlock())
        self.MiddleFlow = Sequential(*self.MiddleFlow)
        self.ExitFlow = Sequential(
            SeparableConvBlocks(in_channels=728,out_channels=1024,mid_channels=728),
            SeparableConv(in_channels=1024,out_channels=1536),
            SeparableConv(in_channels=1536,out_channels=2048),
            AvgPool2d(kernel_size=(10,10)),
            Flatten(),
            FullConnectLayer(in_features=2048,out_features=output_size,output_layer=True)
        )

    def forward(self,x):
        x = self.EntryFlow(x)
        x = self.MiddleFlow(x)
        x = self.ExitFlow(x)
        return x

if __name__ == "__main__":
    from torchsummary import summary

    data = torch.randn((16,3,299,299))
    net = Xception()
    summary(net.cuda(),(3,299,299))