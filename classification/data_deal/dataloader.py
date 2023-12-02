import os
import cv2
import torch
import random
import numpy as np
import torchvision
from torchvision.transforms import Compose,ToPILImage,ToTensor,Resize
from torch.utils.data import Dataset,DataLoader
random.seed(2023)

def GetTrainvalDateSet(train_txt,val_txt,input_size):
    train_dataset = MyDataSet(train_txt,input_size)
    val_dataset = MyDataSet(val_txt,input_size)
    return train_dataset,val_dataset

class MyDataSet(Dataset):
    def __init__(self,data_txt,input_size):
        super().__init__()
        self.image_info_list = self.__loaddata__(data_txt=data_txt)
        self.tf = Compose([
            ToPILImage(),
            Resize([input_size,input_size]),
            ToTensor()
        ])
    def __loaddata__(self,data_txt):
        images_info_list = []
        with open(data_txt,"r",encoding="utf-8")as f:
            data = f.readlines()
        for info in data:
            img_path,target = info.split(",")
            images_info_list.append({"image_path":img_path,"target":int(target)})
        return images_info_list
    def __len__(self):
        return len(self.image_info_list)
    
    def __getitem__(self, index):
        img_data = cv2.imdecode(np.fromfile(self.image_info_list[index]["image_path"],dtype=np.uint8),cv2.IMREAD_COLOR)
        img_target = self.image_info_list[index]["target"]

        img_data = self.tf(img_data)
        img_target = torch.tensor(img_target).long()

        return img_data,img_target
if __name__ == "__main__":
    td,vd = GetTrainvalDateSet("data/Classification/dataset_kaggledogvscat/data",224,{"cat":0,"dog":1})
    i,t = td.__getitem__(0)

    print(len(td))
    print(len(vd))