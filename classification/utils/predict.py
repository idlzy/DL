import os
import sys
import time
import yaml
import tqdm
import torch
import datetime
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
cwd_path = os.getcwd()
cwd_dir = os.path.split(cwd_path)[-1]
if cwd_dir != "DL":
    print("please work at directory 'DL' before start training")
    sys.exit()
else:
    sys.path.append(os.path.join(os.getcwd(),"classification"))
from net import *
from data_deal.dataloader import *
import utils

class Predictor:
    def __init__(self,model_path,input_size=224,device=torch.device("cuda" if torch.cuda.is_available()else "cpu")):
        self.device = device
        self.net = torch.load(model_path).to(self.device)
        self.tf = Compose([
            ToPILImage(),
            Resize([input_size,input_size]),
            ToTensor()
        ])

    def predict(self,img_path):
        img = cv2.imdecode(np.fromfile(img_path,dtype=np.uint8),cv2.IMREAD_COLOR)
        img_data = self.tf(img)
        img_data = torch.unsqueeze(img_data,dim=0).to(self.device)
        output = self.net(img_data)
        print(output)

if __name__ =="__main__":
    predictor = Predictor(model_path="logs\\VGGNet\\best_model.pt")
    while 1:
        img_path = input("输入图片地址(输入stop停止程序)>>>")
        if img_path =="stop":
            break
        predictor.predict(img_path)
