import os
import sys

cwd_path = os.getcwd()
cwd_dir = os.path.split(cwd_path)[-1]
if cwd_dir != "DL":
    print("please work at directory 'DL' before start training")
    sys.exit()
else:
    sys.path.append(os.path.join(os.getcwd(),"classification"))
from net.lenet import *
from net.alexnet import *
from net.vggnet import *
from net.baseblock import *
from net.googlenet import *
from net.resnet import *
Net = {"LeNet":LeNet,
       "AlexNet":AlexNet,
       "VGGNet":VGGNet16,
       "Googlenet":Googlenet,
       "ResNet18":ResNet18,
       "ResNet34":ResNet34,
       "ResNet50":ResNet50,
       "ResNet101":ResNet101,
       "ResNet152":ResNet152,
       }