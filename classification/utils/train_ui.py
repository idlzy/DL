import cv2
import cv2
import sys
import time
import torch
from train import Trainer
from PyQt5.QtGui import QPixmap  
from UI.Ui_UiTrain import Ui_MainWindow
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QObject,pyqtSignal,Qt
from PyQt5.QtWidgets import QMainWindow, QApplication,QMessageBox, QWidget,QFileDialog
import threading
from threading import Thread
import tqdm
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class TrainerThread(QObject):
    count_changed = pyqtSignal(list)
    def __init__(self):
        super().__init__()
        self.trainer = None

    def make_trainer(self,data_yaml_path,net_yaml_path):
        self.trainer = Trainer(data_yaml_path,net_yaml_path)

    def train(self):
        start_time = time.time()
        for epoch in range(self.trainer.EPOCH):
            n_epoch = int(100*epoch/self.trainer.EPOCH)
            if not self.trainer.run_flag:
                return
            torch.cuda.empty_cache()
            """ 进行训练 """
            self.trainer.batch_num = 0
            bar = tqdm.tqdm(self.trainer.train_dataloader)  # 设置进度条
            bar.set_description(f"EPOCH {epoch + 1}")
            epoch_loss = 0
            self.trainer.net.train()  # 开启训练模式，梯度可以被计算
            for data, target in bar:
                if not self.trainer.run_flag:
                    return
                self.trainer.batch_num += 1
                
                """ 将数据转移到训练设备上 """
                data = data.to(self.trainer.device)
                target = target.to(self.trainer.device)
                
                
                if "Googlenet" in self.trainer.model_name:
                    """ 获得网络输出 """
                    output_main,output_aux1,output_aux2 = self.trainer.net(data)

                    """ 计算交叉熵损失值 """
                    loss_main = self.trainer.loss_f(output_main, target)
                    loss_aux1 = self.trainer.loss_f(output_aux1, target)
                    loss_aux2 = self.trainer.loss_f(output_aux2, target)
                    loss = loss_main + 0.3*loss_aux1+ 0.3*loss_aux2

                else:
                    """ 获得网络输出 """
                    output = self.trainer.net(data)

                    """ 计算交叉熵损失值 """
                    loss = self.trainer.loss_f(output, target)

                """ 梯度清零 """
                self.trainer.opt.zero_grad()

                """ 误差反向传播 """
                loss.backward()

                """ 更新权重 """
                self.trainer.opt.step()

                """ 进度条显示loss值 """
                epoch_loss += loss.item()
                bar.set_postfix(epoch_loss=epoch_loss, loss=loss.item())
                n_step = int(100*self.trainer.batch_num/len(self.trainer.train_dataloader))
                self.count_changed.emit([epoch,loss.item(),epoch_loss,0,n_epoch,n_step])
                del loss,data,target
            
            train_loss = epoch_loss/len(self.trainer.train_dataset)
            self.trainer.train_loss_list.append(train_loss)
            self.trainer.writer.add_scalar("train_loss", train_loss, epoch + 1)

            """ 进行验证 """
            self.trainer.batch_num = 0
            self.trainer.net.eval()  # 关闭梯度计算
            bar2 = tqdm.tqdm(self.trainer.val_dataloader)  # 设置进度条
            bar2.set_description(f"VAL {epoch + 1}")
            correct = 0
            val_loss = 0
            for data, target in bar2:
                if not self.trainer.run_flag:
                    return
                self.trainer.batch_num += 1
                """ 将数据转移到训练设备上 """
                data = data.to(self.trainer.device)
                target = target.to(self.trainer.device)

                """ 获得网络输出 """
                output = self.trainer.net(data)
                
                """ 计算交叉熵损失值 """
                loss = self.trainer.loss_f(output, target)

                _, pre = torch.max(output, 1)

                """ 计算正确分类的数量 """
                correct = correct + torch.sum(pre == target)
                val_loss += loss.item()
                n_step = int(100*self.trainer.batch_num/len(self.trainer.val_dataloader))
                self.count_changed.emit([epoch,loss.item(),epoch_loss,val_loss,n_epoch,n_step])
                del loss,data,target

            val_loss = val_loss/len(self.trainer.val_dataset)
            self.trainer.val_loss_list.append(val_loss)
            acc = ((correct / len(self.trainer.val_dataset)).item())

            self.trainer.writer.add_scalar("val_acc", acc, epoch + 1)
            self.trainer.writer.add_scalar("val_loss", val_loss / len(self.trainer.val_dataset), epoch + 1)
            print("acc: ", acc)
            if acc > max(self.trainer.acc_list): 
                """当本次验证的准确度大于历史最大时，更新历史最大值，并保存本次模型"""
                torch.save(self.trainer.net, self.trainer.best_model_save_path)
                torch.save(self.trainer.net.state_dict(), self.trainer.best_state_save_path)
                print(f"Have saved the best model with acc of {round(acc, 4) * 100}% to {self.trainer.best_model_save_path} ...")
                self.trainer.best_epoch = epoch
            self.trainer.acc_list.append(acc)

            if self.trainer.EarlyStop and epoch-self.trainer.best_epoch>=self.trainer.EarlyStopEpoch and max(self.trainer.acc_list)>0.9:
                    """ 满足早停条件后进行停止迭代 """
                    print(f"The accuracy has not improved for over {self.trainer.EarlyStopEpoch} epochs, so, early stop now !")
                    break
            
        end_time = time.time()
        self.trainer.train_time = end_time - start_time
        self.trainer.export_infer()
        self.count_changed.emit(["finish"])

class CodeView(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("图像分类训练系统")
        self.data_yaml_path = None
        self.net_yaml_path = None
        self.trainer_thread = TrainerThread()
        self.set_up()

    def set_up(self):
        self.ui.progressBar.setValue(0)
        self.ui.progressBar_2.setValue(0)
        self.ui.exit.clicked.connect(self.exit_ui)
        self.ui.net_cfg.clicked.connect(self.open_net)
        self.ui.data_cfg.clicked.connect(self.open_data)
        self.ui.train.clicked.connect(self.train_btn)
        self.ui.stop_train.clicked.connect(self.close_train_btn)
        self.ui.model_gen.clicked.connect(self.gen_model)
        self.ui.default_cfg.clicked.connect(self.default_cfg_btn)
        self.trainer_thread.count_changed.connect(self.update_info)
        
    def open_data(self):
        filename = QFileDialog.getOpenFileName(None, '数据配置', 'classification/configs/dataset', 'Data yaml(*.yaml)')
        if filename[0]: 
            self.data_yaml_path = filename[0]
            self.ui.checkBox_data.setChecked(True)
            self.ui.checkBox_gen.setChecked(False)
    def open_net(self):
        filename = QFileDialog.getOpenFileName(None, '模型配置', 'classification/configs/net', 'Net yaml(*.yaml)')
        if filename[0]: 
            self.net_yaml_path = filename[0]
            self.ui.checkBox_model.setChecked(True)
            self.ui.checkBox_gen.setChecked(False)
    def gen_model(self):
        widget = QWidget()
        if not self.ui.checkBox_model.isChecked():
            QMessageBox.warning(widget,'警告','未进行模型配置',QMessageBox.Close)
            return
        if not self.ui.checkBox_data.isChecked():
            QMessageBox.warning(widget,'警告','未进行数据配置',QMessageBox.Close)
            return

        self.trainer_thread.make_trainer(self.data_yaml_path,self.net_yaml_path)
        self.ui.label_model_name.setText(self.trainer_thread.trainer.model_name)
        self.ui.label_input_size.setText(f"[{self.trainer_thread.trainer.input_size},{self.trainer_thread.trainer.input_size}]")
        self.ui.label_cls_num.setText(str(self.trainer_thread.trainer.class_num))
        self.ui.label_epochs.setText(str(self.trainer_thread.trainer.EPOCH))
        self.ui.label_batch_size.setText(str(self.trainer_thread.trainer.batch_size))
        self.ui.label_lr.setText(str(self.trainer_thread.trainer.lr))
        self.ui.label_es.setText(str(self.trainer_thread.trainer.EarlyStop))
        self.ui.label_ese.setText(str(self.trainer_thread.trainer.EarlyStopEpoch))
        self.ui.label_device.setText(self.trainer_thread.trainer.device.type)
        self.ui.label_opt.setText(self.trainer_thread.trainer.opt.__class__.__name__)
        self.ui.label_lossfc.setText(self.trainer_thread.trainer.loss_f.__class__.__name__)
        self.ui.checkBox_gen.setChecked(True)
    def default_cfg_btn(self):
        self.data_yaml_path = "classification/configs/dataset/cat_vs_dog.yaml"
        self.net_yaml_path = "classification/configs/net/resnet.yaml"
        self.ui.checkBox_model.setChecked(True)
        self.ui.checkBox_data.setChecked(True)
        self.ui.checkBox_gen.setChecked(False)

    def train_btn(self):
        widget = QWidget()
        if not self.ui.checkBox_gen.isChecked():
            QMessageBox.warning(widget,'警告','未生成模型',QMessageBox.Close)
            return
        self.ui.progressBar.setValue(0)
        self.ui.progressBar_2.setValue(0)
        self.trainer_thread.trainer.run_flag = 1
        Thread(target=self.trainer_thread.train).start()
    
    def close_train_btn(self):
        if isinstance(self.trainer_thread.trainer,Trainer):
            self.trainer_thread.trainer.run_flag = 0

    def update_info(self,info):

        if info[0]=="finish":
            self.ui.progressBar.setValue(100)
            self.ui.progressBar_2.setValue(100)
            widget = QWidget()
            QMessageBox.warning(widget,'提醒','模型训练完毕',QMessageBox.Close)
            return
        
        # 更新标签显示计数值
        epoch = info[0]
        loss = info[1]
        epoch_loss = info[2]
        val_loss = info[3]
        

        n_epoch = info[4]
        n_step = info[5]
        
        self.ui.res_epoch.setAlignment(Qt.AlignCenter)
        self.ui.res_epoch.setText(str(epoch))
        self.ui.res_loss.setText(str(round(loss,2)))
        self.ui.res_eploss.setText(str(round(epoch_loss,2)))
        self.ui.res_valloss.setText(str(round(val_loss,2)))
        self.ui.res_accbest.setText(str(round(max(self.trainer_thread.trainer.acc_list),4)))


        self.ui.progressBar.setValue(n_epoch)
        self.ui.progressBar_2.setValue(n_step)
        

    def exit_ui(self):
        if isinstance(self.trainer_thread.trainer,Trainer):
            self.trainer_thread.trainer.run_flag = 0
        QApplication.quit()
        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWidget = CodeView()
    myWidget.show()
    sys.exit(app.exec_())