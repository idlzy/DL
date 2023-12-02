from net.baseblock import *

class YOLO(nn.Module):
    def __init__(self,input_size=448,class_num=20,B=2,S=7,Softmax=True):
        super().__init__()
        self.C = class_num
        self.B = B
        self.S = S
        self.Softmax = Softmax
        self.ConvBlocks1 = Sequential(
            BasicConv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2),
            MaxPool2d(kernel_size=(2,2),stride=2),
        )
        self.ConvBlocks2 = Sequential(
            BasicConv2d(in_channels=64,out_channels=192,kernel_size=3,stride=1),
            MaxPool2d(kernel_size=(2,2),stride=2),

        )

        self.ConvBlocks3 = Sequential(
            BasicConv2d(in_channels=192,out_channels=128,kernel_size=1,stride=1),
            BasicConv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1),
            BasicConv2d(in_channels=256,out_channels=256,kernel_size=1,stride=1),
            BasicConv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1),
            MaxPool2d(kernel_size=(2,2),stride=2),
        )

        self.ConvBlocks4 = []
        [self.ConvBlocks4.extend([
                BasicConv2d(in_channels=512,out_channels=256,kernel_size=1,stride=1),
                BasicConv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1),
            ]) for i in range(4)]
        self.ConvBlocks4.extend([
            BasicConv2d(in_channels=512,out_channels=256,kernel_size=1,stride=1),
            BasicConv2d(in_channels=256,out_channels=1024,kernel_size=3,stride=1),
            MaxPool2d(kernel_size=(2,2),stride=2)
        ])
        self.ConvBlocks4 = Sequential(*self.ConvBlocks4)
        
        self.ConvBlocks5 = []
        [self.ConvBlocks5.extend([
                BasicConv2d(in_channels=1024,out_channels=512,kernel_size=1,stride=1),
                BasicConv2d(in_channels=512,out_channels=1024,kernel_size=3,stride=1),
            ]) for i in range(2)]
        self.ConvBlocks5.extend([
            BasicConv2d(in_channels=1024,out_channels=1024,kernel_size=3,stride=1),
            MaxPool2d(kernel_size=(3,3),stride=2,padding=1)
        ])
        self.ConvBlocks5 = Sequential(*self.ConvBlocks5)        
        
        self.ConvBlocks6 = Sequential(
            BasicConv2d(in_channels=1024,out_channels=1024,kernel_size=3,stride=1),
            BasicConv2d(in_channels=1024,out_channels=1024,kernel_size=3,stride=1)
        )

        self.FullConnectLayers = Sequential(
            Flatten(),
            FullConnectLayer(in_features=7*7*1024,out_features=4096),
            FullConnectLayer(in_features=4096,out_features=self.S*self.S*(5*self.B+self.C)),
        )


    def forward(self,x):
        x = self.ConvBlocks1(x)
        x = self.ConvBlocks2(x)
        x = self.ConvBlocks3(x)
        x = self.ConvBlocks4(x)
        x = self.ConvBlocks5(x)
        x = self.ConvBlocks6(x)
        x = self.FullConnectLayers(x)
        x = torch.reshape(x,[-1,self.S,self.S,self.B*5+self.C])

        if self.Softmax:
            x_c = x[:,:,:,0:self.B]
            x_xywh = x[:,:,:,self.B:5*self.B]
            x_cls_obj = x[:,:,:,self.B*5:]
            x_c = torch.sigmoid(x_c)
            x_cls_obj = torch.softmax(x_cls_obj,dim=-1)
            x = torch.cat([x_c,x_xywh,x_cls_obj],dim=-1)
        return x
    

if __name__ == "__main__":
    net = YOLO(class_num=20)
    data = torch.randn((1,3,448,448))
    out = net(data)
    print(out[0,0,0,:])