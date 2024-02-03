import torch
import torch.nn as nn



class YOLOV1Loss(nn.Module):
    def __init__(self,coord=5,noobj=0.5,S=7,B=2,C=20,device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__()
        self.coord = coord
        self.noobj = noobj
        self.S=S
        self.B=B
        self.C=C
        self.device = device
    def forward(self,input,target):
        batch_n = input.shape[0]
        loss_all = torch.tensor(0.).to(self.device)
        for batch in range(batch_n):
            loss_batch = torch.tensor(0.).to(self.device)
            true_label = target[batch]
            pred_label = input[batch]
            loss_xy = torch.tensor(0.).to(self.device)
            loss_wh = torch.tensor(0.).to(self.device)
            loss_objc = torch.tensor(0.).to(self.device)
            loss_noobjc = torch.tensor(0.).to(self.device)
            loss_cls = torch.tensor(0.).to(self.device)
            for i in range(self.S):
                for j in range(self.S):
                    cls_true = true_label[i,j,self.B*5:]
                    cls_pred = pred_label[i,j,self.B*5:]
                    for k in range(self.B):
                        objc_true = true_label[i,j,k]
                        objc_pred = pred_label[i,j,k]
                        if true_label[i,j,k]==1:
                            x_true,y_true,w_true,h_true = true_label[i,j,self.B+4*k:self.B+4*(k+1)]
                            x_pred,y_pred,w_pred,h_pred = pred_label[i,j,self.B+4*k:self.B+4*(k+1)]
                            """目标中心坐标损失"""
                            loss_xy += torch.sum(torch.pow(torch.tensor([x_pred-x_true,y_pred-y_true]), 2))
                            """目标长宽损失"""
                            loss_wh += torch.sum(torch.pow(torch.tensor([torch.sqrt(w_pred)-torch.sqrt(w_true),torch.sqrt(h_pred)-torch.sqrt(h_true)]), 2))
                            """目标置信度损失"""
                            loss_objc += torch.pow(objc_pred-objc_true,2)
                            """目标类别损失"""
                            loss_cls += torch.sum(torch.pow(cls_pred-cls_true,2))
                            break
                    else:
                        """无目标置信度损失"""
                        for k in range(self.B):
                            objc_true = true_label[i,j,k]
                            objc_pred = pred_label[i,j,k]
                            loss_noobjc += torch.pow(objc_pred-objc_true,2)
            loss_batch = self.coord*(loss_xy+loss_wh)+self.noobj*loss_noobjc+loss_objc+loss_cls
            loss_all += loss_batch
        loss_all = loss_all/batch_n
        return loss_all


if __name__ == "__main__":
    input = torch.zeros((1,7,7,30))
    target = torch.zeros((1,7,7,30))
    target[0,3,3,:6] = torch.tensor([1,0,0.5,0.5,0.3,0.3])
    input[0,3,3,:6] = torch.tensor([1,0,0.5,0.5,0.2,0.4])
    loss_fc = YOLOV1Loss()
    loss = loss_fc(input,target)
    print(loss)