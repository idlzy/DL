import os
import sys
import time
import yaml
import tqdm
import torch
import datetime
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
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
from configs.info import log_trian_info




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="hello")
    parser.add_argument('-d','--datayaml',default="cat_vs_dog.yaml",type=str, help='input the data yaml file name')
    parser.add_argument('-n','--netyaml',default="resnet.yaml",type=str,help='input the net yaml file name')

    parser.add_argument('-o','--own',default="false",choices=["true","false"] ,type=str,help='input the net yaml file name')

    opt = parser.parse_args()

    cfg_data_file = opt.datayaml
    cfg_net_file = opt.netyaml

    if opt.own=="false":
        cfg_data_path = os.path.join("classification/configs/data",cfg_data_file)
        cfg_net_path = os.path.join("classification/configs/net",cfg_net_file)
    else:
        cfg_data_path = cfg_data_file
        cfg_net_path = cfg_net_file
        
    with open(cfg_data_path,"r") as f:
        print(f"load {cfg_data_file}......")
        data_cfg = yaml.safe_load(f)
        time.sleep(0.5)
    with open(cfg_net_path,"r") as f:
        print(f"load {cfg_net_file}......")
        net_cfg = yaml.safe_load(f)
        time.sleep(0.5)
    """=========================  配置信息  ========================="""
    utils.seed_everything()
    time_txt_path = "logs/time.txt"
    train_txt = os.path.join(os.path.join(data_cfg["BaseDir"],data_cfg["TrainvalDir"]),"train.txt")
    val_txt = os.path.join(os.path.join(data_cfg["BaseDir"],data_cfg["TrainvalDir"]),"val.txt")
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
    input_size = net_cfg["input_size"]          # 输入图像尺寸
    class_num = net_cfg["class_num"]            # 输出数据维度
    EPOCH = net_cfg["EPOCH"]                    # 迭代次数
    batch_size = net_cfg["batch_size"]          # 批处理数量
    batch_size_val = net_cfg["batch_size_val"]  # 验证集的批处理数量
    lr = net_cfg["lr"]                          # 初始学习率
    EarlyStop = net_cfg["EarlyStop"]            # 是否采用早停策略
    EarlyStopEpoch = net_cfg["EarlyStopEpoch"]  # 15个epoch后acc还没有提升则停止训练
    """=============================================================="""

    """=========================  配置数据集  ========================="""
    train_dataset,val_dataset =  GetTrainvalDateSet(train_txt,val_txt,input_size)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=data_cfg["train_num_workers"])
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,num_workers=data_cfg["val_num_workers"])
    """==============================================================="""

    """=========================  训练配置  ========================="""
    writer = SummaryWriter(logs_save_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设置模型训练设备
    net = Net[model_name](output_size=class_num).to(device)                # 实例化网络，并将网络转移到主训练设备
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

    acc_list = [0]
    train_loss_list = []
    val_loss_list = []
    best_epoch = 0
    time.sleep(1)
    start_time = time.time()
    for epoch in range(EPOCH):
        torch.cuda.empty_cache()
        """ 进行训练 """
        bar = tqdm.tqdm(train_dataloader)  # 设置进度条
        bar.set_description(f"EPOCH {epoch + 1}")
        epoch_loss = 0
        net.train()  # 开启训练模式，梯度可以被计算
        for data, target in bar:
            """ 将数据转移到训练设备上 """
            data = data.to(device)
            target = target.to(device)

            
            if "Googlenet" in model_name:
                """ 获得网络输出 """
                output_main,output_aux1,output_aux2 = net(data)

                """ 计算交叉熵损失值 """
                loss_main = loss_f(output_main, target)
                loss_aux1 = loss_f(output_aux1, target)
                loss_aux2 = loss_f(output_aux2, target)
                loss = loss_main + 0.3*loss_aux1+ 0.3*loss_aux2

            else:
                """ 获得网络输出 """
                output = net(data)

                """ 计算交叉熵损失值 """
                loss = loss_f(output, target)

            """ 梯度清零 """
            opt.zero_grad()

            """ 误差反向传播 """
            loss.backward()

            """ 更新权重 """
            opt.step()

            """ 进度条显示loss值 """
            epoch_loss += loss.item()
            bar.set_postfix(epoch_loss=epoch_loss, loss=loss.item())
            del loss,data,target
        train_loss = epoch_loss/len(train_dataset)
        train_loss_list.append(train_loss)
        writer.add_scalar("train_loss", train_loss, epoch + 1)
        """ 进行验证 """
        net.eval()  # 关闭梯度计算
        bar2 = tqdm.tqdm(val_dataloader)  # 设置进度条
        bar2.set_description(f"VAL {epoch + 1}")
        correct = 0
        val_loss = 0
        for data, target in bar2:
            """ 将数据转移到训练设备上 """
            data = data.to(device)
            target = target.to(device)

            """ 获得网络输出 """
            output = net(data)
            
            """ 计算交叉熵损失值 """
            loss = loss_f(output, target)

            _, pre = torch.max(output, 1)

            """ 计算正确分类的数量 """
            correct = correct + torch.sum(pre == target)
            val_loss += loss.item()
            del loss,data,target

        val_loss = val_loss/len(val_dataset)
        val_loss_list.append(val_loss)
        acc = ((correct / len(val_dataset)).item())

        writer.add_scalar("val_acc", acc, epoch + 1)
        writer.add_scalar("val_loss", val_loss / len(val_dataset), epoch + 1)
        print("acc: ", acc)
        if acc > max(acc_list): 
            """当本次验证的准确度大于历史最大时，更新历史最大值，并保存本次模型"""
            torch.save(net, best_model_save_path)
            torch.save(net.state_dict(), best_state_save_path)
            print(f"Have saved the best model with acc of {round(acc, 4) * 100}% to {best_model_save_path} ...")
            best_epoch = epoch
        acc_list.append(acc)
        if EarlyStop and epoch-best_epoch>=EarlyStopEpoch and max(acc_list)>0.9:
                """ 满足早停条件后进行停止迭代 """
                print(f"The accuracy has not improved for over {EarlyStopEpoch} epochs, so, early stop now !")
                break
    writer.close()
    end_time = time.time()
    last_time = end_time - start_time
    with open(time_txt_path, "a", encoding="utf-8") as f:
        text = f"使用模型为{model_name},训练完成于{datetime.datetime.now()}, 用时{last_time}s,最高准确度为{round(max(acc_list), 4) * 100}%\n"
        f.write(text)