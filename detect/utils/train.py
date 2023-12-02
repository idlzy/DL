import os
import sys
import time
import yaml
import tqdm
import torch
import datetime
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
if __name__ == "__main__":
    cwd_path = os.getcwd()
    cwd_dir = os.path.split(cwd_path)[-1]
    if cwd_dir != "DL":
        print("please work at directory 'DL' before start training")
        sys.exit()
    else:
        sys.path.append(os.path.join(os.getcwd(),"detect"))
    from net import *
    from data_deal.dataloader import *
    from configs.info import log_trian_info

    cfg_data_file = "voc.yaml"
    cfg_net_file = "yolo.yaml"

    cfg_data_path = os.path.join("detect/configs/data",cfg_data_file)
    cfg_net_path = os.path.join("detect/configs/net",cfg_net_file)


    with open(cfg_data_path,"r") as f:
        data_cfg = yaml.safe_load(f)

    with open(cfg_net_path,"r") as f:
        net_cfg = yaml.safe_load(f)


    """=========================  配置信息  ========================="""
    time_txt_path = "logs/time.txt"
    data_dir = data_cfg["VOCDIR"]
    model_name = net_cfg["model_name"]
    logs_save_path = f"{net_cfg['logs_save_path']}/{model_name}"
    best_model_save_path = os.path.join(logs_save_path, "best_model.pt")
    best_state_save_path = os.path.join(logs_save_path, "best_state.pth")
    if not os.path.exists(logs_save_path):
        os.makedirs(logs_save_path)
    if not os.path.exists(time_txt_path):
        file = open(time_txt_path, "w")
        file.close()
    """============================================================="""

    """=========================  配置超参数  ========================="""
    split_rate=data_cfg["split_rate"]
    input_size = net_cfg["input_size"]          # 输入图像尺寸
    class_num = net_cfg["class_num"]            # 检测的类别数量
    EPOCH = net_cfg["EPOCH"]                    # 迭代次数
    batch_size = net_cfg["batch_size"]          # 批处理数量
    batch_size_val = net_cfg["batch_size_val"]  # 验证集的批处理数量
    lr = net_cfg["lr"]                          # 初始学习率
    EarlyStop = net_cfg["EarlyStop"]            # 是否采用早停策略
    EarlyStopEpoch = net_cfg["EarlyStopEpoch"]  # 15个epoch后acc还没有提升则停止训练
    """=============================================================="""

    """=========================  配置数据集  ========================="""
    train_dataset,val_dataset =  GetTrainvalDateSet(data_dir=data_dir,input_size=input_size,split_rate=split_rate)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=data_cfg["train_num_workers"])
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,num_workers=data_cfg["val_num_workers"])
    """==============================================================="""

    """=========================  训练配置  ========================="""
    writer = SummaryWriter(logs_save_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设置模型训练设备
    net = Net[model_name](class_num=class_num).to(device)                                     # 实例化Lenet网络，并将网络转移到主训练设备
    loss_f = torch.nn.CrossEntropyLoss()                                   # 定义损失函数为交叉熵损失函数
    opt = torch.optim.SGD(net.parameters(), lr=lr)                         # 采用SGD优化策略
    """=============================================================="""

    """=========================  输出信息  ========================="""
    log_trian_info({"model name":model_name,
                    "input size":f"[{input_size},{input_size}]",
                    "class num":class_num,
                    "batch size":batch_size,
                    "learing rate": lr,
                    "early stop":EarlyStop,
                    "early stop epoch":EarlyStopEpoch,
                    "device":device.type,
                    "optimizer":opt.__class__.__name__,
                    "loss function":loss_f.__class__.__name__,
                    })
    """=============================================================="""



