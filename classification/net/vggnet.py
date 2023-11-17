from net.baseblock import *
from torchsummary import summary
class VGGblock(nn.Module):
    def __init__(self,in_channels,out_channels,n):
        super().__init__()
        self.Convblock = Sequential()
        self.Convblock.add_module("Conv1",ConvLayer(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,downsample=False))
        for i in range(n-1):
            self.Convblock.add_module(f"Conv{i+2}",ConvLayer(in_channels=out_channels,out_channels=out_channels,kernel_size=3,stride=1,downsample=False))        
        self.Convblock.add_module("MaxPooling",MaxPool2d(kernel_size=(2,2)))
    
    def forward(self,x):
        x = self.Convblock(x)
        return x
        
class VGGNet16(nn.Module):
    def __init__(self,output_size=1000):
        super().__init__()
        self.ConvBlocks = Sequential(
            VGGblock(in_channels=3,out_channels=64,n=2),
            VGGblock(in_channels=64,out_channels=128,n=2),
            VGGblock(in_channels=128,out_channels=256,n=3),
            VGGblock(in_channels=256,out_channels=512,n=3),
            VGGblock(in_channels=512,out_channels=512,n=3),
            Flatten()
        )
        self.FullConnectLayers = Sequential(
            FullConnectLayer(in_features=25088,out_features=4096,drop_p=0.5),
            FullConnectLayer(in_features=4096,out_features=4096,drop_p=0.5),
            FullConnectLayer(in_features=4096,out_features=output_size,output_layer=True)
        )
    def forward(self,x):
        x = self.ConvBlocks(x)
        x = self.FullConnectLayers(x)
        return x


if __name__ == "__main__":
    net = VGGNet16()
    # for name,param in net.named_parameters():
    #     print(name)
    # data = torch.randn((16,3,224,224))
    # out = net(data)
    # print(out.shape)
    summary(net.cuda(), (3, 224, 224))