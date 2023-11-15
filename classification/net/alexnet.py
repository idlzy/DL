from baseblock import *

class AlexNet(nn.Module):
    def __init__(self, input_size=224,output_size=1000):
        super().__init__()
        self.ConvLayer = Sequential(
            ConvLayer(in_channels=3,out_channels=96,kernel_size=11,stride=4,padding=3,downsample=False),
            ConvLayer(in_channels=96,out_channels=256,kernel_size=5,stride=1),
            ConvLayer(in_channels=256,out_channels=384,kernel_size=3,stride=1),
            ConvLayer(in_channels=384,out_channels=384,kernel_size=3,stride=1,downsample=False),
            ConvLayer(in_channels=384,out_channels=256,kernel_size=3,stride=1),
            Flatten()
        )

        self.FullConnectLayer = Sequential(
            FullConnectLayer(in_features=6*6*256,out_features=4096,drop_p=0.5),
            FullConnectLayer(in_features=4096,out_features=4096,drop_p=0.5),
            FullConnectLayer(in_features=4096,out_features=output_size,drop_p=0,softmax=True),
        )


    def forward(self,x):
        x = self.ConvLayer(x)
        x = self.FullConnectLayer(x)
        return x
    
if __name__ == "__main__":
    net = AlexNet()
    for name,param in net.named_parameters():
        print(name)
    data = torch.randn((16,3,224,224))
    out = net(data)
    print(out.shape)
    