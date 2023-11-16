from net.baseblock import *


class LeNet(nn.Module):
    def __init__(self,input_size=32,output_size=10):
        super().__init__()

        self.ConvLayer = Sequential(
            ConvLayer(in_channels=3,out_channels=6,kernel_size=5,stride=1,padding=0),
            ConvLayer(in_channels=6,out_channels=16,kernel_size=5,stride=1,padding=0),
            Flatten()
        )

        self.FullConnectLayer = Sequential(
            FullConnectLayer(in_features=(int(((input_size-4)/2-4)/2)**2)*16,out_features=120),
            FullConnectLayer(in_features=120,out_features=84),
            FullConnectLayer(in_features=84,out_features=output_size,output_layer=True)
        )
    def forward(self,x):
        x = self.ConvLayer(x)
        x = self.FullConnectLayer(x)
        return x
    
if __name__ == "__main__":
    net = LeNet(input_size=28)
    for name,param in net.named_parameters():
        print(name)
    data = torch.randn((16,3,28,28))
    out = net(data)
    print(out.shape)
    