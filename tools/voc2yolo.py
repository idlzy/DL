import xml.etree.ElementTree as ET
import os
import glob
import argparse
import yaml
import tqdm


# voc标注的目标框坐标值转换到yolo标注的目标框坐标值的函数
def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

class VOC2YOLO():
    def __init__(self,voc_folder,yolo_folder,label2num_dic):
        self.voc_folder = voc_folder
        self.yolo_folder = yolo_folder
        self.label2num_dic = label2num_dic
        if not os.path.exists(yolo_folder):
            os.makedirs(yolo_folder)

    # 对单个voc标注文件进行转换成其对应的yolo文件的函数
    def convert_annotation(self,xml_file):
        file_name = xml_file.strip(".xml")  # 这一步将所有voc格式标注文件取出后缀名“.xml”，方便接下来作为yolo格式标注文件的名称
        in_file = open(os.path.join(self.voc_folder, xml_file))  # 打开当前转换的voc标注文件
        out_file = open(os.path.join(self.yolo_folder, file_name + ".txt", ), 'w')  # 创建并打开要转换成的yolo格式标注文件
        tree = ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        for obj in root.iter('object'):
            cls = obj.find('name').text
            cls_id = self.label2num_dic[cls]
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text),
                float(xmlbox.find('xmax').text),
                float(xmlbox.find('ymin').text),
                float(xmlbox.find('ymax').text))
            bb = convert((w, h), b)
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
    
    def run(self):
        xml_fileList = os.listdir(self.voc_folder)  # 将所有voc格式的标注文件的名称取出存放到列表xml_fileList中
        for xml_file in tqdm.tqdm(xml_fileList):  # 这里的for循环开始依次对所有voc格式标注文件生成其对应的yolo格式的标注文件
            self.convert_annotation(xml_file)
        num2label = {value:key for key,value in self.label2num_dic.items()}
        class_txt = os.path.join(self.yolo_folder,"classes.txt")
        with open(class_txt,"w",encoding='utf-8')as f:
            for i in range(len(num2label)):
                f.write(num2label[i]+'\n')
if __name__ == "__main__":       
    parser = argparse.ArgumentParser(description="hello")
    parser.add_argument('-d','--datayaml',default="detect/configs/data/voc.yaml",type=str, help='input the data yaml file name')
    opt = parser.parse_args()
    with open(opt.datayaml,'r')as f:
        data_cfg = yaml.safe_load(f)
    voc_folder = os.path.join(data_cfg['VOCDIR'],"Annotations")
    yolo_folder = os.path.join(data_cfg['VOCDIR'],"labels")
    label2num_dic = data_cfg["class_dic"]
    woker = VOC2YOLO(voc_folder=voc_folder,yolo_folder=yolo_folder,label2num_dic=label2num_dic)
    woker.run()