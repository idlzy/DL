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


class Trainer:
    def __init__(self,cfg_data_path,cfg_net_path):
        with open(cfg_data_path,"r") as f:
            print(f"load {os.path.splitext(cfg_data_path)[0]}......")
            self.data_cfg = yaml.safe_load(f)
            time.sleep(0.5)
        with open(cfg_net_path,"r") as f:
            print(f"load {os.path.splitext(cfg_net_path)[0]}......")
            self.net_cfg = yaml.safe_load(f)
            time.sleep(0.5)
        """=========================  配置信息  ========================="""
        utils.seed_everything()
        self.time_txt_path = "logs/time.txt"
        train_txt = os.path.join(os.path.join(self.data_cfg["BaseDir"],self.data_cfg["TrainvalDir"]),"train.txt")
        val_txt = os.path.join(os.path.join(self.data_cfg["BaseDir"],self.data_cfg["TrainvalDir"]),"val.txt")
        self.model_name = self.net_cfg["model_name"]

        self.logs_save_path = f"{self.net_cfg['logs_save_path']}/{utils.get_log_sub_dir(self.net_cfg['logs_save_path'])}/{self.model_name}"
        self.best_model_save_path = os.path.join(self.logs_save_path, "best_model.pt")
        self.best_state_save_path = os.path.join(self.logs_save_path, "best_state.pth")
        self.save_infer_path = os.path.join(self.logs_save_path,"infer.yaml")
        if not os.path.exists(self.logs_save_path):
            os.makedirs(self.logs_save_path)
        if not os.path.exists(self.time_txt_path):
            file = open(self.time_txt_path, "w")
            file.close()
        """============================================================="""

        """=========================  配置超参数  ========================="""
        self.input_size = self.net_cfg["input_size"]          # 输入图像尺寸
        self.class_num = self.net_cfg["class_num"]            # 输出数据维度
        self.EPOCH = self.net_cfg["EPOCH"]                    # 迭代次数
        self.batch_size = self.net_cfg["batch_size"]          # 批处理数量
        self.batch_size_val = self.net_cfg["batch_size_val"]  # 验证集的批处理数量
        self.lr = self.net_cfg["lr"]                          # 初始学习率
        self.EarlyStop = self.net_cfg["EarlyStop"]            # 是否采用早停策略
        self.EarlyStopEpoch = self.net_cfg["EarlyStopEpoch"]  # 15个epoch后acc还没有提升则停止训练
        """=============================================================="""

        """=========================  配置数据集  ========================="""
        self.train_dataset,self.val_dataset =  GetTrainvalDateSet(train_txt,val_txt,self.input_size)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,num_workers=self.data_cfg["train_num_workers"])
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True,num_workers=self.data_cfg["val_num_workers"])
        """==============================================================="""

        """=========================  训练配置  ========================="""
        self.writer = SummaryWriter(self.logs_save_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设置模型训练设备
        self.net = Net[self.model_name](output_size=self.class_num).to(self.device)                # 实例化网络，并将网络转移到主训练设备
        self.loss_f = torch.nn.CrossEntropyLoss()                                   # 定义损失函数为交叉熵损失函数
        self.opt = torch.optim.SGD(self.net.parameters(), lr=self.lr)                         # 采用SGD优化策略
        """=============================================================="""

        """=========================  输出信息  ========================="""
        log_trian_info({"model name":self.model_name,
                        "input size":f"[{self.input_size},{self.input_size}]",
                        "class num":self.class_num,
                        "epochs": self.EPOCH,
                        "batch size":self.batch_size,
                        "learning rate": self.lr,
                        "early stop":self.EarlyStop,
                        "early stop epoch":self.EarlyStopEpoch,
                        "device":self.device.type,
                        "optimizer":self.opt.__class__.__name__,
                        "loss function":self.loss_f.__class__.__name__,
                        })
        """=============================================================="""


        """=========================  指标保存  ========================="""
        self.acc_list = [0]              # 用于保存准确率
        self.train_loss_list = []        # 用于保存训练集损失
        self.val_loss_list = []          # 保存验证集损失
        self.best_epoch = 0              # 记录准确率表现最好的一次epoch
        self.batch_num = 0               # 记录batch数
        self.train_time = 0              # 记录模型训练时间
        self.run_flag = 1                # 记录训练标识
        """=============================================================="""


    def train_epoch(self,epoch):    
        torch.cuda.empty_cache()
        """ 进行训练 """
        self.batch_num = 0
        bar = tqdm.tqdm(self.train_dataloader)  # 设置进度条
        bar.set_description(f"EPOCH {epoch + 1}")
        epoch_loss = 0
        self.net.train()  # 开启训练模式，梯度可以被计算
        for data, target in bar:
            if not self.run_flag:
                return
            self.batch_num += 1

            """ 将数据转移到训练设备上 """
            data = data.to(self.device)
            target = target.to(self.device)
            
            
            if "Googlenet" in self.model_name:
                """ 获得网络输出 """
                output_main,output_aux1,output_aux2 = self.net(data)

                """ 计算交叉熵损失值 """
                loss_main = self.loss_f(output_main, target)
                loss_aux1 = self.loss_f(output_aux1, target)
                loss_aux2 = self.loss_f(output_aux2, target)
                loss = loss_main + 0.3*loss_aux1+ 0.3*loss_aux2

            else:
                """ 获得网络输出 """
                output = self.net(data)

                """ 计算交叉熵损失值 """
                loss = self.loss_f(output, target)

            """ 梯度清零 """
            self.opt.zero_grad()

            """ 误差反向传播 """
            loss.backward()

            """ 更新权重 """
            self.opt.step()

            """ 进度条显示loss值 """
            epoch_loss += loss.item()
            bar.set_postfix(epoch_loss=epoch_loss, loss=loss.item())
            del loss,data,target
        train_loss = epoch_loss/len(self.train_dataset)
        self.train_loss_list.append(train_loss)
        self.writer.add_scalar("train_loss", train_loss, epoch + 1)
        """ 进行验证 """
        self.batch_num = 0
        self.net.eval()  # 关闭梯度计算
        bar2 = tqdm.tqdm(self.val_dataloader)  # 设置进度条
        bar2.set_description(f"VAL {epoch + 1}")
        correct = 0
        val_loss = 0
        for data, target in bar2:
            self.batch_num += 1
            """ 将数据转移到训练设备上 """
            data = data.to(self.device)
            target = target.to(self.device)

            """ 获得网络输出 """
            output = self.net(data)
            
            """ 计算交叉熵损失值 """
            loss = self.loss_f(output, target)

            _, pre = torch.max(output, 1)

            """ 计算正确分类的数量 """
            correct = correct + torch.sum(pre == target)
            val_loss += loss.item()
            del loss,data,target

        val_loss = val_loss/len(self.val_dataset)
        self.val_loss_list.append(val_loss)
        acc = ((correct / len(self.val_dataset)).item())

        self.writer.add_scalar("val_acc", acc, epoch + 1)
        self.writer.add_scalar("val_loss", val_loss / len(self.val_dataset), epoch + 1)
        print("acc: ", acc)
        if acc > max(self.acc_list): 
            """当本次验证的准确度大于历史最大时，更新历史最大值，并保存本次模型"""
            torch.save(self.net, self.best_model_save_path)
            torch.save(self.net.state_dict(), self.best_state_save_path)
            print(f"Have saved the best model with acc of {round(acc, 4) * 100}% to {self.best_model_save_path} ...")
            self.best_epoch = epoch
        self.acc_list.append(acc)
        
    def train(self):
        
        time.sleep(1)
        start_time = time.time()
        
        for epoch in range(self.EPOCH):
            self.train_epoch(epoch)
            if self.EarlyStop and epoch-self.best_epoch>=self.EarlyStopEpoch and max(self.acc_list)>0.9:
                    """ 满足早停条件后进行停止迭代 """
                    print(f"The accuracy has not improved for over {self.EarlyStopEpoch} epochs, so, early stop now !")
                    break
        
        end_time = time.time()
        self.train_time = end_time - start_time
        self.export_infer()
    
    
    def export_infer(self):
        self.writer.close()
        with open(self.time_txt_path, "a", encoding="utf-8") as f:
            text = f"使用模型为{self.model_name},训练完成于{datetime.datetime.now()}, 用时{self.train_time}s,最高准确度为{round(max(self.acc_list), 4) * 100}%\n"
            f.write(text)
        """
        to save infer yaml which you can use when infer.
        """
        label2num_dic = self.data_cfg["class_dic"]
        num2label_dic = {value:key for key,value in label2num_dic.items()}
        save_infer_data = {
            "model_path":os.path.join(cwd_path,self.best_model_save_path),
            "num2label":num2label_dic,
            "input_size":self.input_size
        }
        with open(self.save_infer_path,"w",encoding="utf-8") as f:
            yaml.dump(save_infer_data,f)
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="hello")
    parser.add_argument('-d','--datayaml',default="cat_vs_dog.yaml",type=str, help='input the data yaml file name')
    parser.add_argument('-n','--netyaml',default="resnet.yaml",type=str,help='input the net yaml file name')

    parser.add_argument('-o','--own',default="false",choices=["true","false"] ,type=str,help='input the net yaml file name')

    option = parser.parse_args()

    cfg_data_file = option.datayaml
    cfg_net_file = option.netyaml

    if option.own=="false":
        cfg_data_path = os.path.join("classification/configs/dataset",cfg_data_file)
        cfg_net_path = os.path.join("classification/configs/net",cfg_net_file)
    else:
        cfg_data_path = cfg_data_file
        cfg_net_path = cfg_net_file
        
    trainer = Trainer(cfg_data_path,cfg_net_path)
    trainer.train()