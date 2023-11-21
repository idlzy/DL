from net.baseblock import *


class InceptionBlock(nn.Module):
    def __init__(self,in_channels,out_conv1x1,out_conv3x3_r,out_conv3x3,out_conv5x5_r,out_conv5x5,out_poolproj):
        super().__init__()
        self.Conv1x1Branch = BasicConv2d(in_channels=in_channels,out_channels=out_conv1x1,kernel_size=1,stride=1)
        self.Conv3x3Branch = Sequential(
            BasicConv2d(in_channels=in_channels,out_channels=out_conv3x3_r,kernel_size=3,stride=1),
            BasicConv2d(in_channels=out_conv3x3_r,out_channels=out_conv3x3,kernel_size=3,stride=1)
        )
        self.Conv5x5Branch = Sequential(
            BasicConv2d(in_channels=in_channels,out_channels=out_conv5x5_r,kernel_size=3,stride=1),
            BasicConv2d(in_channels=out_conv5x5_r,out_channels=out_conv5x5,kernel_size=3,stride=1)
        )
        self.PoolBranch = Sequential(
            MaxPool2d(kernel_size=(3,3),stride=1,padding=1),
            BasicConv2d(in_channels=in_channels,out_channels=out_poolproj,kernel_size=1,stride=1)
        )

    def forward(self,x):
        x1 = self.Conv1x1Branch(x) 
        # print("x1:",x1.shape)
        x2 = self.Conv3x3Branch(x)
        # print("x2:",x2.shape)
        x3 = self.Conv5x5Branch(x)
        # print("x3:",x3.shape)
        x4 = self.PoolBranch(x)
        # print("x4:",x4.shape)
        x = torch.cat([x1,x2,x3,x4],dim=1)
        return x   

class InceptionAux(nn.Module):
    def __init__(self,in_channels,output_size=1000, input_size=14):
        super().__init__()
        self.Aux = Sequential(
            AvgPool2d(kernel_size=(5,5),stride=3),
            BasicConv2d(in_channels=in_channels,out_channels=128,kernel_size=1,stride=1),
            Flatten(),
            FullConnectLayer(in_features=128*((input_size-2)//3)**2,out_features=1024,drop_p=0.7),
            FullConnectLayer(in_features=1024,out_features=output_size,output_layer=True),
        )

    def forward(self,x):
        x = self.Aux(x)
        return x

class Googlenet(nn.Module):
    def __init__(self,input_size=224,output_size=1000):
        super().__init__()
        self.maxpool = MaxPool2d(kernel_size=(3,3),stride=2,padding=1)

        self.droplayer = Dropout(0.4)
        self.ConvHead = Sequential(
            BasicConv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2),
            MaxPool2d(kernel_size=(3,3),stride=2,padding=1),
            BatchNorm2d(num_features=64),
            BasicConv2d(in_channels=64,out_channels=64,kernel_size=1,stride=1),
            BasicConv2d(in_channels=64,out_channels=192,kernel_size=3,stride=1),
            BatchNorm2d(num_features=192),
            MaxPool2d(kernel_size=(3,3),stride=2,padding=1)
        )
        self.incept3a = InceptionBlock(in_channels=192,out_conv1x1=64,out_conv3x3_r=96,out_conv3x3=128,out_conv5x5_r=16,out_conv5x5=32,out_poolproj=32)
        self.incept3b = InceptionBlock(in_channels=256,out_conv1x1=128,out_conv3x3_r=128,out_conv3x3=192,out_conv5x5_r=32,out_conv5x5=96,out_poolproj=64)
        self.incept4a = InceptionBlock(in_channels=480,out_conv1x1=192,out_conv3x3_r=96,out_conv3x3=208,out_conv5x5_r=16,out_conv5x5=48,out_poolproj=64)
        self.aux4a = InceptionAux(in_channels=512,output_size=output_size)
        self.incept4b = InceptionBlock(in_channels=512,out_conv1x1=160,out_conv3x3_r=112,out_conv3x3=224,out_conv5x5_r=24,out_conv5x5=64,out_poolproj=64)
        self.incept4c = InceptionBlock(in_channels=512,out_conv1x1=128,out_conv3x3_r=128,out_conv3x3=256,out_conv5x5_r=24,out_conv5x5=64,out_poolproj=64)
        self.incept4d = InceptionBlock(in_channels=512,out_conv1x1=112,out_conv3x3_r=144,out_conv3x3=288,out_conv5x5_r=32,out_conv5x5=64,out_poolproj=64)
        self.aux4d = InceptionAux(in_channels=528,output_size=output_size)
        self.incept4e = InceptionBlock(in_channels=528,out_conv1x1=256,out_conv3x3_r=160,out_conv3x3=320,out_conv5x5_r=32,out_conv5x5=128,out_poolproj=128)

        self.incept5a = InceptionBlock(in_channels=832,out_conv1x1=256,out_conv3x3_r=160,out_conv3x3=320,out_conv5x5_r=32,out_conv5x5=128,out_poolproj=128)
        self.incept5b = InceptionBlock(in_channels=832,out_conv1x1=384,out_conv3x3_r=192,out_conv3x3=384,out_conv5x5_r=48,out_conv5x5=128,out_poolproj=128)
        
        self.FC = Sequential(
            AvgPool2d(kernel_size=7,stride=1),
            Flatten(),
            Dropout(0.4),
            FullConnectLayer(in_features=1024,out_features=output_size,output_layer=True)
        )


    def forward(self,x):
        x = self.ConvHead(x)
        x = self.incept3a(x)
        x = self.incept3b(x)
        x = self.maxpool(x)

        x = self.incept4a(x)
        if self.training:
            x_4aout = self.aux4a(x)

        x = self.incept4b(x)
        x = self.incept4c(x)
       
        x = self.incept4d(x)
        if self.training:
            x_4dout = self.aux4d(x)

        x = self.incept4e(x)
        x = self.maxpool(x) 
        x = self.incept5a(x)
        x = self.incept5b(x)
        x = self.FC(x)

        if self.training:
            return x,x_4aout,x_4dout
        return x

if __name__ == "__main__":
    in_ch = 3
    input_size = 224
    data = torch.randn((1,in_ch,input_size,input_size))
    net = Googlenet(output_size=2)
    net.eval()
    out = net(data)
    print(out)
    # from torchsummary import summary
    # summary(net.cuda(),(3,224,224),batch_size=16)
    # for name,param in net.named_parameters():
    #     print(name)