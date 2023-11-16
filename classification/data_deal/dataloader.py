import os
import cv2
import torch
import random
import numpy as np
import torchvision
from torchvision.transforms import Compose,ToPILImage,ToTensor,Resize
from torch.utils.data import Dataset,DataLoader
random.seed(2023)

def GetTrainvalDateSet(data_dir,input_size,split_rate=0.7):
    labels_list = os.listdir(data_dir)
    image_info_list = []
    label2num_dic = {i:labels_list.index(i) for i in labels_list}
    num2label_dic = {value:key for key,value in label2num_dic.items()}
    for label_name in labels_list:
        images_path = os.path.join(data_dir,label_name)
        images_name_list = os.listdir(images_path)
        for image_name in images_name_list:
            image_path = os.path.join(images_path,image_name)
            image_info_list.append({"image_path":image_path,"target":label2num_dic[label_name]})
    random.shuffle(image_info_list)
    train_end_idx = int(len(image_info_list)*split_rate)
    train_dataset = MyDataSet(image_info_list[:train_end_idx],input_size=input_size)
    val_dataset = MyDataSet(image_info_list[train_end_idx:],input_size=input_size)
    return train_dataset,val_dataset

class MyDataSet(Dataset):
    def __init__(self,image_info_list,input_size):
        super().__init__()
        self.image_info_list = image_info_list
        self.tf = Compose([
            ToPILImage(),
            Resize([input_size,input_size]),
            ToTensor()
        ])

    def __len__(self):
        return len(self.image_info_list)
    
    def __getitem__(self, index):
        img_data = cv2.imdecode(np.fromfile(self.image_info_list[index]["image_path"],dtype=np.uint8),cv2.IMREAD_COLOR)
        img_target = self.image_info_list[index]["target"]

        img_data = self.tf(img_data)
        img_target = torch.tensor(img_target).long()

        return img_data,img_target
if __name__ == "__main__":
    td,vd = GetTrainvalDateSet("data/Classification/dataset_kaggledogvscat",224)
    i,t = td.__getitem__(0)

    print(len(td))
    print(len(vd))