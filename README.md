## 深度学习领域经典模型算法复现
---

## 目录
1. [图像分类模型 Images Classification](#图像分类模型)
2. [目标检测模型 Object Detection](#目标检测模型)
3. [相关文件下载 Related File Download](#相关文件下载)
4. [所需环境 Environment](#所需环境)
5. [训练步骤 Train Model](#训练步骤)
6. [预测推理 Predict and Infer](#预测推理)
7. [作者 Author](#作者)

## 图像分类模型
- [x] LeNet
- [x] AlexNet
- [x] VGGNet
- [x] Googlenet
- [ ] ResNet
- [ ] DenseNet
- [ ] MoblieNetv1
- [ ] Xception
- [ ] MoblieNetv2
- [ ] ShuffleNet
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

## 相关文件下载
### 数据集下载
在分类任务里，数据集来自kaggle的catvsdog数据集
该数据集链接：https://pan.baidu.com/s/11slBJvYNfctIehGWlRPGYQ 
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
│  │      ├─cat
│  │      └─dog
```

#### 生成类别字典(*windows系统下路径中使用'\\',而Linux系统下路径使用'/'*)
```shell
python tools/generate_classdic.py -f data/Classification/dataset_kaggledogvscat/ -s catdog.yaml
```


首先确保我们的工作路径需要在DL路径下
然在我们需要修改对应任务里的configs文件夹下的数据集配置文件如classification/configs/data/cat_vs_dog.yaml文件，设置自己的数据集的data_path和所需分类的类别，类别数量较少时，手动填写即可，当类别很多时，可以使用根目录下tools里的generate_classdic.py脚本来一键生成类别字典
接着配置模型的超参数配置文件，如classification/configs/net/alexnet.yaml里，可设置batch_size,epoch,lr等超参数
最后，在classification/utils/train.py里修改cfg_data_file变量和cfg_net_file，更改成你需要的配置文件的名称。修改后了之后，直接在终端运行以下命令即可

```shell
python classification/utils/train.py
```





## 预测推理

## 作者
**Name**:
<font color="blue">Ricardo</font><br><br>
**Email**:
<font color="green">1437633423@qq.com</font><br><br>
如有问题可邮箱留言。