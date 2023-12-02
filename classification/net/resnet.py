from net.baseblock import *

class Identity(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1):
        super().__init__()
        self.Conv1x1 = Sequential(
            BasicConv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,stride=stride,bias=False),
            BatchNorm2d(num_features=out_channels)
        )
        self.Equ = True if in_channels==out_channels else False

    def forward(self,x):
        if not self.Equ:
            x = self.Conv1x1(x)
        return x

class ResidualStructBase(nn.Module):
    def __init__(self,in_channels, out_channels,stride=1):
        super().__init__()
        self.ConvBlocks = Sequential(
            BasicConv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=stride,bias=False),
            BatchNorm2d(num_features=out_channels),
            BasicConv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,stride=1,bias=False),
            BatchNorm2d(num_features=out_channels)
        )
        self.identity = Identity(in_channels=in_channels,out_channels=out_channels,stride=stride)
    def forward(self,x):    
        x1 = self.ConvBlocks(x)
        x2 = self.identity(x)
        x = x1 + x2 
        return x

class ResidualStructNeck(nn.Module):
    def __init__(self,in_channels, out_channels,stride=1):
        super().__init__()
        self.NeckChannels = out_channels//4
        self.ConvBlocks = Sequential(
            BasicConv2d(in_channels=in_channels,out_channels=self.NeckChannels,kernel_size=1,stride=1,bias=False),
            BatchNorm2d(num_features=self.NeckChannels),
            BasicConv2d(in_channels=self.NeckChannels,out_channels=self.NeckChannels,kernel_size=3,stride=stride,bias=False),
            BatchNorm2d(num_features=self.NeckChannels),
            BasicConv2d(in_channels=self.NeckChannels,out_channels=out_channels,kernel_size=1,stride=1,bias=False),
            BatchNorm2d(num_features=out_channels),
        )
        self.identity = Identity(in_channels=in_channels,out_channels=out_channels,stride=stride)
    def forward(self,x):    
        x1 = self.ConvBlocks(x)
        x2 = self.identity(x)
        x = x1 + x2 
        return x

class ResidualBlocks(nn.Module):
    def __init__(self,in_channels,out_channels,n,dowsample=True,Deep=False):
        super().__init__()
        self.Blocks = Sequential()
        ResidualStruct = ResidualStructNeck if Deep else ResidualStructBase
        for i in range(n):
            if i==0 :
                if dowsample:
                    self.Blocks.add_module(f"Conv{i+1}",ResidualStruct(in_channels=in_channels,out_channels=out_channels,stride=2))
                else:
                    self.Blocks.add_module(f"Conv{i+1}",ResidualStruct(in_channels=in_channels,out_channels=out_channels,stride=1))
            else:
                self.Blocks.add_module(f"Conv{i+1}",ResidualStruct(in_channels=out_channels,out_channels=out_channels,stride=1))
    def forward(self,x):
        x = self.Blocks(x) 
        return x


class ResNet(nn.Module):
    def __init__(self,input_size=224,output_size=1000,residual_blocks=[3,4,6,3],Deep=True):
        super().__init__()
        self.Conv1 = Sequential(
            BasicConv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2,bias=False),
            BatchNorm2d(num_features=64)
        ) 
        
        self.Maxpool = MaxPool2d(kernel_size=(3,3),stride=2,padding=1)
        self.last_channels = 512
        if not Deep:
            self.Conv2 = ResidualBlocks(in_channels=64,out_channels=64,n=residual_blocks[0],dowsample=False,Deep=Deep)
            self.Conv3 = ResidualBlocks(in_channels=64,out_channels=128,n=residual_blocks[1],Deep=Deep)
            self.Conv4 = ResidualBlocks(in_channels=128,out_channels=256,n=residual_blocks[2],Deep=Deep)
            self.Conv5 = ResidualBlocks(in_channels=256,out_channels=512,n=residual_blocks[3],Deep=Deep)
        else:
            self.Conv2 = ResidualBlocks(in_channels=64,out_channels=256,n=residual_blocks[0],dowsample=False,Deep=Deep)
            self.Conv3 = ResidualBlocks(in_channels=256,out_channels=512,n=residual_blocks[1],Deep=Deep)
            self.Conv4 = ResidualBlocks(in_channels=512,out_channels=1024,n=residual_blocks[2],Deep=Deep)
            self.Conv5 = ResidualBlocks(in_channels=1024,out_channels=2048,n=residual_blocks[3],Deep=Deep)
            self.last_channels = 2048
        self.avgpool = AvgPool2d(kernel_size=(7,7),stride=1)
        self.FullConnectLayers = Sequential(
            Flatten(),
            FullConnectLayer(in_features=self.last_channels,out_features=output_size,output_layer=True)
        )
        
        
    def forward(self,x):
        x = self.Conv1(x)
        x = self.Maxpool(x)
        x = self.Conv2(x)
        x = self.Conv3(x)
        x = self.Conv4(x)
        x = self.Conv5(x)
        x = self.avgpool(x)
        x = self.FullConnectLayers(x)
        return x

class ResNet18(ResNet):
    def __init__(self,input_size=224,output_size=1000,residual_blocks=[2,2,2,2],Deep=False):
        super().__init__(input_size=input_size,output_size=output_size,residual_blocks=residual_blocks,Deep=Deep)

class ResNet34(ResNet):
    def __init__(self,input_size=224,output_size=1000,residual_blocks=[3,4,6,3],Deep=False):
        super().__init__(input_size=input_size,output_size=output_size,residual_blocks=residual_blocks,Deep=Deep)


class ResNet50(ResNet):
    def __init__(self,input_size=224,output_size=1000,residual_blocks=[3,4,6,3],Deep=True):
        super().__init__(input_size=input_size,output_size=output_size,residual_blocks=residual_blocks,Deep=Deep)

class ResNet101(ResNet):
    def __init__(self,input_size=224,output_size=1000,residual_blocks=[3,4,23,3],Deep=True):
        super().__init__(input_size=input_size,output_size=output_size,residual_blocks=residual_blocks,Deep=Deep)

class ResNet152(ResNet):
    def __init__(self,input_size=224,output_size=1000,residual_blocks=[3,8,36,3],Deep=True):
        super().__init__(input_size=input_size,output_size=output_size,residual_blocks=residual_blocks,Deep=Deep)

if __name__ == "__main__":
    data = torch.randn((16, 3, 224, 224))
    net = ResNet(residual_blocks=[3,4,6,3],Deep=False)
    out = net(data)
    from torchsummary import summary
    summary(net.cuda(),(3,224,224))
    # count = 0
    # for name,_ in net.named_parameters():
    #     if "Conv2d" in name and "weight" in name and "identity" not in name:
    #         print(name)
    #         count +=1
    # print(count)
    # import torchvision
    # from torchvision.models import resnet50
    # r_net = resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
    # summary(r_net.cuda(),(3,224,224))