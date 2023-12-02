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
    train_val_info_path = os.path.join(data_dir,"train_val_txt")
    if not os.path.exists(train_val_info_path):
        os.makedirs(train_val_info_path)
    images_dir = os.path.join(data_dir,"JPEGImages")
    image_name_list = os.listdir(images_dir)
    for image_name in image_name_list:
        name = image_name.split(".")[0]
        image_path = os.path.join(images_dir,image_name)
        


class MyDataSet(Dataset):
    def __init__(self):
        super().__init__()
        pass

    def __len__(self):
        pass
    
    def __getitem__(self, index):
        pass
    
if __name__ == "__main__":
    GetTrainvalDateSet("data/ObjectDetection/VOC2007",input_size=448)
    print("hello")