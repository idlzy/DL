import os
import sys

cwd_path = os.getcwd()
cwd_dir = os.path.split(cwd_path)[-1]
if cwd_dir != "DL":
    print("please work at directory 'DL' before start training")
    sys.exit()
else:
    sys.path.append(os.path.join(os.getcwd(),"classification"))
from net.lenet import LeNet
from net.alexnet import AlexNet
from net.vggnet import VGGNet16
from net.googlenet import Googlenet
from net.resnet import ResNet18,ResNet34,ResNet50,ResNet101,ResNet152
from net.mobilenet import MobileNet
from net.xception import Xception
from net.densenet import DenseNet121,DenseNet169,DenseNet201,DenseNet264
Net = {"LeNet":LeNet,
       "AlexNet":AlexNet,
       "VGGNet":VGGNet16,
       "Googlenet":Googlenet,
       "ResNet18":ResNet18,
       "ResNet34":ResNet34,
       "ResNet50":ResNet50,
       "ResNet101":ResNet101,
       "ResNet152":ResNet152,
       "MobileNet":MobileNet,
       "Xception":Xception,
       "DenseNet121":DenseNet121,
       "DenseNet169":DenseNet169,
       "DenseNet201":DenseNet201,
       "DenseNet264":DenseNet264,
       }