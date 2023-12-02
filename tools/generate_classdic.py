import os
import sys
import yaml
import time
import tqdm
import argparse
import lxml.etree as ET
cwd_path = os.getcwd()
cwd_dir = os.path.split(cwd_path)[-1]
if cwd_dir != "DL":
    print("please work at directory 'DL' before start training")
    sys.exit()

wastebin_dir = "wastebin"
if not os.path.exists(wastebin_dir):
    os.makedirs(wastebin_dir)

"""============== SafeDumperWithOrder =============="""
class SafeDumperWithOrder(yaml.SafeDumper):
    pass
def dict_representer(dumper, data):
    return dumper.represent_mapping(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, data.items())
SafeDumperWithOrder.add_representer(dict, dict_representer)
"""================================================="""

def dump_yaml(opt):
    labels_list = []
    if opt.mode == "voc":
        print("数据检索中......")
        time.sleep(0.1)
        ana_dir = os.path.join(opt.filedir,"Annotations")
        ana_name_list = os.listdir(ana_dir)
        for name in tqdm.tqdm(ana_name_list):
            ana_path = os.path.join(ana_dir,name)
            tree = ET.ElementTree(file=ana_path)  # 替换成你的XML文件路径
            node_object = tree.findall("object")
            for object in node_object:
                node_name = object.find("name")
                name_text = node_name.text
                if name_text not in labels_list:
                    labels_list.append(name_text)

    elif opt.mode == "coco":
        print("暂不支持coco数据格式.可以选择将其转换为voc格式后再运行该脚本")
        return 0
    else:
        labels_list = os.listdir(opt.filedir)
    
    label2num_dic = {i:labels_list.index(i) for i in labels_list}
    num2label_dic = {value:key for key,value in label2num_dic.items()}
    data_label2num = {"class_dic":label2num_dic}
    data_num2label = {"class_dic":num2label_dic}

    print(data_label2num)
    with open(os.path.join(wastebin_dir,opt.savename),"w",encoding="utf-8") as f:
        if opt.label2num=="true":
            yaml.dump(data_label2num,f,Dumper=SafeDumperWithOrder)
        else:
            yaml.dump(data_num2label,f,Dumper=SafeDumperWithOrder)
    print(f"finished {opt.savename} generating, please go to wastebin to take a look")




parser = argparse.ArgumentParser(description="hello")
parser.add_argument('-f','--filedir',required=True,type=str, help='input the file name')
parser.add_argument('-l','--label2num',default="true",choices=["true","false"],type=str, help='select label2num or num2label dic')
parser.add_argument('-s','--savename',default="out.yaml",type=str,help='the output file name')
parser.add_argument('-m','--mode',default="none",choices=["voc","coco","none"],type=str,help='the mode of data')
opt = parser.parse_args()

if __name__ == "__main__":
    dump_yaml(opt)