import os
import sys
import yaml
import time
import tqdm
import random
import argparse
import numpy as np
import lxml.etree as ET
from collections import Counter
from voc2yolo import convert
random.seed(2023)


def get_annotation_yolo(txt_file,class_dic):
    class_dic_inv = {value:key for key,value in class_dic.items()}
    
    label = ''
    cls_list = []
    with open(txt_file,'r') as f:
        raw_label_list = f.readlines()
    
    for raw_label in raw_label_list:
        cls_id = int(raw_label.split(' ')[0])
        cls_list.append(class_dic_inv[cls_id])
        raw_label = raw_label.strip('\n')
        
        label += f',{raw_label}'
    return label,cls_list

def get_annotation_voc(xml_file,class_dic):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    res_list = []
    label = ''
    cls_list = []
    for obj in root.iter('object'):
        cls = obj.find('name').text
        cls_list.append(cls)
        cls_id = class_dic[cls]
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text),
             float(xmlbox.find('xmax').text),
             float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        res_list.append([cls_id,*bb])
    for res in res_list:
        label+=f",{res[0]} {res[1]} {res[2]} {res[3]} {res[4]}"
    return label,cls_list


def generate_datainfo(opt):
    print("数据信息统计中......")
    image_info_list = []
    res_list = []
    time.sleep(0.5)
    with open(opt.yamlfile,"r") as f:
        print(f"load {opt.yamlfile}......")
        data_cfg = yaml.safe_load(f)
        time.sleep(0.5)
    if opt.mode == "coco":
        print("coco格式的数据集处理等待开放中,可将改写成voc格式......")
    elif opt.mode == "yolo":
        data_dir = data_cfg["VOCDIR"]
        label2num_dic = data_cfg["class_dic"]
        split_rate = data_cfg["split_rate"]
        images_dir = os.path.join(data_dir,"JPEGImages")
        labels_dir = os.path.join(data_dir,"labels")
        images_name_list = os.listdir(images_dir)
        for image_name in tqdm.tqdm(images_name_list):
            image_id,_ = os.path.splitext(image_name)
            image_path = os.path.join(images_dir,image_name)
            label_txt_path = os.path.join(labels_dir,image_id+".txt")
            label,cls_list = get_annotation_yolo(label_txt_path,label2num_dic)
            image_info_list.append(image_path+label+"\n")
            res_list.extend(cls_list)
        train_val_info_path = os.path.join(data_dir,data_cfg["TrainvalDir"])

    elif opt.mode == "voc":
        data_dir = data_cfg["VOCDIR"]
        label2num_dic = data_cfg["class_dic"]
        split_rate = data_cfg["split_rate"]
        images_dir = os.path.join(data_dir,"JPEGImages")
        labels_dir = os.path.join(data_dir,"Annotations")
        images_name_list = os.listdir(images_dir)
        for image_name in tqdm.tqdm(images_name_list):
            image_id,_ = os.path.splitext(image_name)
            image_path = os.path.join(images_dir,image_name)
            label_xml_path = os.path.join(labels_dir,image_id+".xml")
            label,cls_list = get_annotation_voc(label_xml_path,label2num_dic)
            image_info_list.append(image_path+label+"\n")
            res_list.extend(cls_list)
        train_val_info_path = os.path.join(data_dir,data_cfg["TrainvalDir"])
        
    else:
        data_dir = os.path.join(data_cfg["BaseDir"],data_cfg["DataDir"])
        label2num_dic = data_cfg["class_dic"]
        split_rate = data_cfg["split_rate"]
        labels_list = os.listdir(data_dir)
        for label_name in labels_list:
            images_path = os.path.join(data_dir,label_name)
            images_name_list = os.listdir(images_path)
            for image_name in images_name_list:
                res_list.append(label_name)
                image_path = os.path.join(images_path,image_name)
                image_info_list.append(f"{image_path},{label2num_dic[label_name]}\n")
        train_val_info_path = os.path.join(data_cfg["BaseDir"],data_cfg["TrainvalDir"])
        
    random.shuffle(image_info_list)
    res_dic = Counter(res_list)
    print("+"+20 * "—" +"+"+20* "—"+"+")
    print("|"f"{'class':^20}|{'number':^20}|")
    print("+"+20 * "—" +"+"+20* "—"+"+")
    for label_name in label2num_dic.keys():
        print("|"f"{label_name:^20}|{res_dic[label_name]:^20}|")
    print("+"+20 * "—" +"+"+20* "—"+"+")
    train_end_idx = int(len(image_info_list)*split_rate)
    if not os.path.exists(train_val_info_path):
        os.makedirs(train_val_info_path)
    train_txt_path = os.path.join(train_val_info_path,"train.txt")
    val_txt_path = os.path.join(train_val_info_path,"val.txt")
    with open(train_txt_path,"w",encoding="utf-8")as f:
        f.writelines(image_info_list[:train_end_idx])
    with open(val_txt_path,"w",encoding="utf-8")as f:
        f.writelines(image_info_list[train_end_idx:])
    print(f"生成训练数据集图像路径和对应标签信息至{train_txt_path},共{len(image_info_list[:train_end_idx])}条样本")
    print(f"生成验证数据集图像路径和对应标签信息至{val_txt_path},共{len(image_info_list[train_end_idx:])}条样本")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="hello")
    parser.add_argument('-y','--yamlfile',default=r"detect\configs\dataset\voc.yaml",type=str, help='input the yaml file name')
    parser.add_argument('-m','--mode',default="yolo",choices=["voc","yolo","coco","none"],type=str,help='the mode of data')
    opt = parser.parse_args()
    generate_datainfo(opt)
    


# convert_annotation(r"C:\Users\14376\Desktop\DL\data\ObjectDetection\VOC2007\Annotations\000002.xml",data_cfg["class_dic"])





