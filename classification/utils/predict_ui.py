import cv2
import cv2
import sys
import time
import torch
from predict import get_Predictor
from PyQt5.QtGui import QPixmap  
from UI.Ui_UiPredict import Ui_MainWindow
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication,QMessageBox, QWidget,QFileDialog


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
class CodeView(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("图像分类识别系统")
        self.img_path = None
        self.Predictor = None
        self.set_up()

    def set_up(self):
        self.ui.openfile.clicked.connect(self.open_file)
        self.ui.cfginfer.clicked.connect(self.cfg_infer)
        self.ui.detect.clicked.connect(self.detect)
        self.ui.exit.clicked.connect(self.exit_ui)
        

    def cfg_infer(self):
        filename = QFileDialog.getOpenFileName(None, '推理配置', '.', 'Infer yaml(*.yaml)')
        if filename[0]: 
            self.Predictor = get_Predictor(filename[0])
            self.ui.checkBox_model.setChecked(True)

    def detect(self):
        widget = QWidget()
        if not self.ui.checkBox_model.isChecked():
            QMessageBox.warning(widget,'警告','未加载模型！',QMessageBox.Close)
            return
        if not self.ui.checkBox_pic.isChecked():
            QMessageBox.warning(widget,'警告','未加载图片！',QMessageBox.Close)
            return
    
        res = self.Predictor.predict(self.img_path)
        self.ui.outres.setText(res)

    def open_file(self):
        filename = QFileDialog.getOpenFileName(None, '选择图片', '.', 'Image files(*.jpg *.gif *.png)')
        if filename[0]:  
            self.ui.show.setPixmap(QPixmap(filename[0]))  
            self.img_path = filename[0]
            self.ui.outres.setText("")
            self.ui.checkBox_pic.setChecked(True)
    def exit_ui(self):
        QApplication.quit()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWidget = CodeView()
    myWidget.show()
    sys.exit(app.exec_())