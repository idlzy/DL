import os
import sys
import time
import yaml
import tqdm
import torch
import argparse
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
    def __init__(self,model_path,num2label,input_size=224,device=torch.device("cuda" if torch.cuda.is_available()else "cpu")):
        self.device = device
        self.net = torch.load(model_path).to(self.device)
        self.tf = Compose([
            ToPILImage(),
            Resize([input_size,input_size]),
            ToTensor()
        ])
        self.num2label = num2label
    def predict(self,img_path):
        img = cv2.imdecode(np.fromfile(img_path,dtype=np.uint8),cv2.IMREAD_COLOR)
        img_data = self.tf(img)
        img_data = torch.unsqueeze(img_data,dim=0).to(self.device)
        output = self.net(img_data)
        _, pre = torch.max(output, 1)
        num = pre.item()
        res = self.num2label[num]
        print(res)
        return res

def get_Predictor(infer):
    with open(infer,"r") as f:
        infer_cfg = yaml.safe_load(f)
    num2label_dic = infer_cfg["num2label"]
    model_path = infer_cfg["model_path"]
    predictor = Predictor(model_path=model_path,num2label=num2label_dic)
    return predictor


if __name__ =="__main__":
    parser = argparse.ArgumentParser(description="hello")
    parser.add_argument('-i','--inferfile',required=True, type=str,help='set the data yaml which has been used when training')
    parser.add_argument('-s','--source',required=True, type=str,help='set the image path')
    opt = parser.parse_args()
    
    predictor = get_Predictor(opt.inferfile)
    predictor.predict(opt.source)
