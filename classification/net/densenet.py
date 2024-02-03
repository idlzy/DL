from net.baseblock import *

class BN_ReLU_Conv(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride):
        super().__init__()
        padding = (kernel_size-1)//2
        self.model = Sequential(
            BatchNorm2d(num_features=in_channels),
            Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding,bias=False),
            ReLU(inplace=True)
        )
    def forward(self,x):
        x = self.model(x)
        return x


class TransitionLayer(nn.Module):
    def __init__(self,num_features,out_features):
        super().__init__()
        self.model = Sequential(
            BN_ReLU_Conv(in_channels=num_features,out_channels=out_features,kernel_size=1,stride=1),
            AvgPool2d(kernel_size=(2,2),stride=2)
        )
    def forward(self,x):
        x = self.model(x)
        return x
    
class Bottlenecklayer(nn.Module):
    def __init__(self,in_channels,grow_rate):
        super().__init__()
        self.model = Sequential(
            BN_ReLU_Conv(in_channels=in_channels,out_channels=grow_rate*4,kernel_size=1,stride=1),
            BN_ReLU_Conv(in_channels=grow_rate*4,out_channels=grow_rate,kernel_size=3,stride=1)
        )

    def forward(self,x):
        x_out = self.model(x)
        x_out = torch.concat([x,x_out],dim=1)
        return x_out


class DenseBlock(nn.Module):
    def __init__(self,in_channels,n,grow_rate=32):
        super().__init__()
        self.Blocks = Sequential()
        for i in range(n):
            self.Blocks.add_module(f"Block layer_{i+1}",Bottlenecklayer(in_channels=in_channels+grow_rate*i,grow_rate=grow_rate))
    def forward(self,x):
        x = self.Blocks(x)
        return x

class DenseNet(nn.Module):
    def __init__(self,output_size=1000,dense_blocks=[6,12,24,16],grow_rate=32,theta_C=0.5):
        super().__init__()
        block_init_channels = grow_rate*2
        self.HeadConv = Sequential(
            BN_ReLU_Conv(in_channels=3,out_channels=block_init_channels,kernel_size=7,stride=2),
            MaxPool2d(kernel_size=(3,3),stride=2,padding=1),
        )

        self.DenseBlocks = Sequential()
        num_features = block_init_channels
        for i in range(len(dense_blocks)):
            self.DenseBlocks.add_module(f"DenseBlock_{i}", DenseBlock(in_channels=num_features,n=dense_blocks[i]))
            num_features += grow_rate*dense_blocks[i]
            if i!=len(dense_blocks)-1:
                self.DenseBlocks.add_module(f"TransitionLayer_{i}", TransitionLayer(num_features=num_features,out_features=int(theta_C*num_features)))
                num_features = int(theta_C*num_features)
        self.classifier = Sequential(
            AvgPool2d(kernel_size=(7,7),stride=1),
            Flatten(),
            FullConnectLayer(in_features=num_features,out_features=output_size,drop_p=0.2,output_layer=True)
        )
    def forward(self,x):
        x = self.HeadConv(x)
        x = self.DenseBlocks(x)
        x = self.classifier(x)
        return x


class DenseNet121(DenseNet):
    def __init__(self, output_size=1000, dense_blocks=[6, 12, 24, 16], grow_rate=32, theta_C=0.5):
        super().__init__(output_size, dense_blocks, grow_rate, theta_C)

class DenseNet169(DenseNet):
    def __init__(self, output_size=1000, dense_blocks=[6, 12, 32, 32], grow_rate=32, theta_C=0.5):
        super().__init__(output_size, dense_blocks, grow_rate, theta_C)

class DenseNet201(DenseNet):
    def __init__(self, output_size=1000, dense_blocks=[6, 12, 48, 32], grow_rate=32, theta_C=0.5):
        super().__init__(output_size, dense_blocks, grow_rate, theta_C)

class DenseNet264(DenseNet):
    def __init__(self, output_size=1000, dense_blocks=[6, 12, 64, 48], grow_rate=32, theta_C=0.5):
        super().__init__(output_size, dense_blocks, grow_rate, theta_C)



if __name__ == '__main__':
    # in_ch = 32
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    data = torch.randn(1,3,224,224)
    net = DenseNet121()
    # out = net(data)
    # print(out.shape)

    net = net.to(device)
    from torchsummary import summary
    summary(net,(3,224,224))