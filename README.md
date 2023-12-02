## 深度学习领域经典模型算法复现
---

## 目录
1. [项目介绍 Introduction](#项目介绍)
2. [图像分类模型 Images Classification](#图像分类模型)
3. [目标检测模型 Object Detection](#目标检测模型)
4. [图像分割模型 Image Segmentation](#图像分割模型)
5. [相关文件下载 Related File Download](#相关文件下载)
6. [所需环境 Environment](#所需环境)
7. [训练步骤 Train Model](#训练步骤)
8. [预测推理 Predict and Infer](#预测推理)
9. [作者 Author](#作者)
10. [项目计划表 Project Schedule](#项目计划表)

## 项目介绍
DL为Deep learning的简称。深度学习近年来在计算机视觉，自然语言处理，推荐系统等领域取得了很多成果，解决了很多复杂的模式识别难题，使得人工智能相关技术取得了很大的进步。因此，为了学习和保存这些先进技术，我们开源了DL这个项目。该项目将复现深度学习计算机视觉领域诸多优秀算法，供大家学习交流。

## 图像分类模型
- [x] LeNet
- [x] AlexNet
- [x] VGGNet
- [x] Googlenet
- [x] ResNet
- [ ] MoblieNetv1
- [ ] Xception
- [ ] DenseNet
- [ ] ShuffleNet
- [ ] MoblieNetv2
- [ ] EfficientNet
- [ ] EfficientNetv2
<p>更新中....</p>

## 目标检测模型
- [ ] R-CNN
- [ ] OverFeat
- [ ] Fast-RCNN
- [ ] YOLOv1
- [ ] YOLOv2
- [ ] SSD
- [ ] Faster-RCNN
- [ ] Mask-RCNN
- [ ] YOLOv3
- [ ] RetinaNet
- [ ] YOLOv4
<p>更新中....</p>

## 图像分割模型
- [ ] U-Net
- [ ] U-Net++
<p>更新中....</p>

## 相关文件下载
### 数据集下载
在分类任务里，数据集来自kaggle的catvsdog数据集
该数据集链接：https://pan.baidu.com/s/1dSPTRYY54aYSmVTdFQ-ZPg 
提取码：2580


## 所需环境
```python
numpy==1.24.4
opencv_contrib_python==4.8.1.78
torch==2.1.0+cu121
tqdm==4.66.1
```


## 训练步骤
### 1. 图像分类
#### 将仓库克隆到本地
```shell
git clone https://github.com/idlzy/DL.git
cd DL
```
#### 在项目根目录下创建存放存放数据集的文件夹
```shell
mkdir -p data/Classification
```
#### 将数据集解压好放置在上一步创建的文件夹下，其文件结构如下
```
├─data
│  ├─Classification
│  │  └─dataset_kaggledogvscat
│  │      ├─data
│  │      │  ├─cat
│  │      │  └─dog
│  │      └─train_val_info
```
在分类任务里，数据集放在data下，同级的train_val_info在数据集对应的yaml文件中被指定(可参考cat_vs_dog.yaml)，用于存放包含训练集和验证集的图像路径和图像标签，目录里的文件为txt文件类型。

#### 生成类别字典(*windows系统下路径中使用'\\',而Linux系统下路径使用'/'*)
```shell
python tools/generate_classdic.py -f data/Classification/dataset_kaggledogvscat/data -s catdog.yaml
```

生成的类别字典将输入到-s选项所指定的文件里，然后我们需要将生成的类别字典复制到我们的data配置信息里，如文件classification\configs\data\cat_vs_dog.yaml
```yaml
BaseDir: "data/Classification/dataset_kaggledogvscat"
DataDir: "data"
TrainvalDir: "train_val_info"          
split_rate: 0.8
train_num_workers : 2
val_num_workers : 2

class_dic:
  cat: 0
  dog: 1
```
其中split_rate表示训练集占全部数据的比例，其他部分为验证集
train_num_workers 和 val_num_works 分别表示 训练集、验证集的数据读取线程数。
DataDir表示的目录里存放每个类别的图片
<br><br>

#### 生成训练集和验证集

```shell
python tools/generate_trainval.py -y classification/configs/data/cat_vs_dog.yaml -m none
```
其中 -y 选项为已经写好了类别字典的数据集配置文件，-m 为数据格式。 默认为none，表示是分类数据集，其他可选项有voc、coco等
划分好了的训练集和验证集将存入使用的数据集配置文件里TrainvalDir所表示的目录下。


在网络的配置文件中，如文件classification\configs\net\alexnet.yaml

```yaml
# config info 
model_name : "ResNet18"               # model_name: you can select from those [RseNet18 ResNet34 ResNet50 ResNet101 ResNet152]
logs_save_path : "logs"
# config hyper-parameters
input_size : 224                      # images input size
class_num : 2                         # output size (class numbers)
EPOCH : 100                           # epoch nums
batch_size : 8                        # batch size numbers
batch_size_val: 4                     # val batch size numbers
lr : 0.1                              # init learing rate
EarlyStop : True                      # Whether to adopt early stop strategy
EarlyStopEpoch : 15                   # Stop training if accuracy has not improved after 15 epochs

```
logs_save_path（相对于项目跟目录）用来指定模型参数和训练指标数据的保存位置
我们可以修改该文件中的超参数来进行模型的调节
<br><br>
最后我们在终端项目根目录下运行如下命令开始训练模型

```shell
python classification/utils/train.py -d cat_vs_dog.yaml -n resnet.yaml
```
这里用-f选项指定数据集配置文件，-n选项指定模型配置文件。需要注意的是。如果配置文件放在了detect/configs/data或detect/configs/net目录下，是不需要写路径的，只需要写文件名称即可，而对于未放在该目录下的配置文件，则需要写为准确的文件路径，并设置-o选项为true，如

```shell
python classification/utils/train.py -d /home/ricardo/cat_vs_dog.yaml -n /home/ricardo/resnet.yaml -o true
```

## 预测推理
下波更新中上线......

## 作者
**Name**:
<font color="blue">Ricardo</font><br><br>
**Email**:
<font color="green">1437633423@qq.com</font><br><br>
如有问题可邮箱留言。

## 项目计划表
- [ ] 在12月22号前，实现YOLO模型
- [ ] 在1月7号前，实现所有目前常见的图像分类模型
- [ ] 在1月15号前，实现图像分割模型U-Net
- [ ] 在2月8号前，实现YOLOv2和YOLOv3模型
- [ ] 在3月1号前，实现YOLOV4
- [ ] 在3月15号前，实现YOLOV5
- [ ] 在3月25号前，实现RCNN
- [ ] 在4月5号前，实现Fast-RCNN
- [ ] 在4月15号前，实现Faster-RCNN
- [ ] 在4月30号前，实现Mask-RCNN
- [ ] 在5月15号前，实现SSD
- [ ] 在6月1号前，实现YOLOx
- [ ] 在6月15号前，实现YOLOv6
- [ ] 在6月30号前，实现YOLOv7
- [ ] 在7月15号前，实现YOLOv8
- [ ] 在7月22号前，实现U-Net++
- [ ] 在9月1号前，项目开启自然语言处理和深度强化学习领域
