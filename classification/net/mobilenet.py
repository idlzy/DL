from net.baseblock import *

class DepthwiseConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1):
        super().__init__()
        self.Conv3x3 = Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=3,stride=stride,padding=1,bias=False,groups=in_channels)
        self.BN1 = BatchNorm2d(num_features=in_channels)
        self.RelU1 = ReLU(inplace=True)
        self.Conv1x1 = Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,bias=False)
        self.BN2 = BatchNorm2d(num_features=out_channels)
        self.RelU2 = ReLU(inplace=True)
    def forward(self,x):
        x = self.Conv3x3(x)
        x = self.BN1(x)
        x = self.RelU1(x)
        x = self.Conv1x1(x)
        x = self.BN2(x)
        x = self.RelU2(x)
        return x

class MobileNet(nn.Module):
    def __init__(self,output_size=1000):
        super().__init__()
        
        loopConvBlocks = []
        for i in range(5):
            loopConvBlocks.append(DepthwiseConvBlock(in_channels=512,out_channels=512,stride=1))
        self.ConvBlocks = Sequential(
            BasicConv2d(in_channels=3,out_channels=32,kernel_size=3,stride=2,padding=1,bias=False),
            BatchNorm2d(num_features=32),
            DepthwiseConvBlock(in_channels=32,out_channels=64,stride=1),
            DepthwiseConvBlock(in_channels=64,out_channels=128,stride=2),
            DepthwiseConvBlock(in_channels=128,out_channels=128,stride=1),
            DepthwiseConvBlock(in_channels=128,out_channels=256,stride=2),
            DepthwiseConvBlock(in_channels=256,out_channels=256,stride=1),
            DepthwiseConvBlock(in_channels=256,out_channels=512,stride=2),
            *loopConvBlocks,
            DepthwiseConvBlock(in_channels=512,out_channels=1024,stride=2),
            DepthwiseConvBlock(in_channels=1024,out_channels=1024,stride=1),
            AvgPool2d(kernel_size=(7,7)),
        )
        self.FullConnectLayers = Sequential(
            Flatten(),
            FullConnectLayer(in_features=1024,out_features=output_size,output_layer=True)
        )

    def forward(self,x):
        x = self.ConvBlocks(x)
        x = self.FullConnectLayers(x)
        return x
    
if __name__ == "__main__":
    from torchsummary import summary
    data = torch.randn((8,3,224,224))
    net = MobileNet(2)
    out = net(data)
    # summary(net.cuda(),(3,224,224))
    print(out.shape)