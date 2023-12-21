import os
import cv2
import torch
import numpy as np
from torchvision.transforms import Compose,ToPILImage,ToTensor,Resize
from torch.utils.data import Dataset


def GetTrainvalDateSet(train_txt,val_txt,input_size):
    train_dataset = MyDataSet(train_txt,input_size)
    val_dataset = MyDataSet(val_txt,input_size)
    return train_dataset,val_dataset

class MyDataSet(Dataset):
    def __init__(self,data_txt,input_size,class_num=20,B=2,S=7):
        super().__init__()
        self.image_info_list = self.__loaddata__(data_txt)
        self.tf = Compose([
            ToPILImage(),
            Resize([input_size,input_size]),
            ToTensor()
        ])
        self.input_size = input_size
        self.C = class_num
        self.B = B
        self.S = S
        self.label_shape = (self.S,self.S,self.B*5+self.C)
    def __loaddata__(self,data_txt):
        with open(data_txt,"r",encoding="utf-8")as f:
            raw_data = f.readlines()
        data_info_list = []
        for rd in raw_data:
            infos = rd.strip("\n").split(',')
            image_path = infos[0]
            label_str_ = infos.copy()
            label_str_.remove(image_path)
            label_ = [list(map(float,i.split(' '))) for i in label_str_]
            label = np.asarray(label_)
            data_info_list.append({"image_path":image_path,"label":label})
        return data_info_list
    
    def __len__(self):
        return len(self.image_info_list)
    
    def __getitem__(self, index):
        image_info = self.image_info_list[index]
        image_path = image_info["image_path"]
        image_label = image_info["label"]
        obj_num = image_label.shape[0]

        img = cv2.imdecode(np.fromfile(image_path,dtype=np.uint8),cv2.IMREAD_COLOR)
        H,W,_ = img.shape
        data = self.tf(img)
        label = np.zeros(self.label_shape)

        # print(image_path)

        for i in range(obj_num):
            cls,x,y,w,h = image_label[i]
            grid_ny = int((y*self.label_shape[0]))
            grid_nx = int((x*self.label_shape[1]))
            grid_y = (y*7)%1
            grid_x = (x*7)%1
            for i in range(self.B):
                label[grid_ny,grid_nx,i]=1
                label[grid_ny,grid_nx,self.B+4*i:self.B+4*(i+1)]=grid_x,grid_y,w,h
                break
            label[grid_ny,grid_nx,self.B*5+int(cls)] = 1
        label = torch.tensor(label)
        return data,label

def decode(label,class_num=20,B=2,S=7):
    detect_info = []
    for i in range(S):
        for j in range(S):
            for k in  range(B):
                if label[i,j,k]==1:
                    x,y,w,h = label[i,j,B+4*k:B+4*(k+1)]
                    cls_array = label[i,j,B*5:]
                    x = (x+j)/S
                    y = (y+i)/S
                    cls_id = np.argmax(cls_array)
                    detect_info.append([cls_id,x,y,w,h])
    return detect_info

def draw_box(img,bbox):
    H,W,_ = img.shape
    for box in bbox:
        cls_id,x,y,w,h = box
        x = W*x
        y = H*y
        w = W*w
        h = H*h
        x_min = int(x-(w/2))
        y_min = int(y-(h/2))
        x_max = int(x+(w/2))
        y_max = int(y+(h/2))
        p1 = (x_min,y_min)
        p2 = (x_max,y_max)
        cv2.rectangle(img,p1,p2,(0,0,255),3)
    return img
if __name__ == "__main__":
    train_dataset = MyDataSet(r"data\ObjectDetection\VOC2007\train_val_info\val.txt",448)
    tar,label = train_dataset.__getitem__(78)
    bbox = decode(label)
    print(len(bbox))
    img_id = '8234'
    img_path = rf"data/ObjectDetection/VOC2007\JPEGImages\00{img_id}.jpg"
    img = cv2.imread(img_path)
    img = draw_box(img,bbox)
    cv2.imshow("img",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()