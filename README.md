## 深度学习领域经典模型算法的复现
- [深度学习领域经典模型算法的复现](#深度学习领域经典模型算法的复现)
- [图像分类模型](#图像分类模型)
- [目标检测模型](#目标检测模型)
- [相关文件下载](#相关文件下载)
- [训练模型](#训练模型)
- [作者](#作者)

## 图像分类模型
- [x] LeNet
- [x] AlexNet
- [ ] VGGNet
- [ ] Googlenet
- [ ] ResNet
- [ ] MoblieNetv1
- [ ] MoblieNetv2
<p>更新中....</p>

## 目标检测模型
- [ ] R-CNN
- [ ] OverFeat
- [ ] YOLOV1
- [ ] YOLOV2
- [ ] SSD
- [ ] Fast-RCNN
- [ ] YOLOV3
- [ ] Faster-RCNN
- [ ] Mask-RCNN
<p>更新中....</p>

## 相关文件下载
在分类任务里，数据集来自kaggle的catvsdog数据集
该数据集链接：https://pan.baidu.com/s/11slBJvYNfctIehGWlRPGYQ 
提取码：2580

## 训练模型
<p>首先我们的工作路径需要在DL路径下</p>
<p>然在我们需要修改对应任务里的configs文件夹下的数据集配置文件如classification/configs/data/cat_vs_dog.yaml文件，设置自己的数据集的data_path</p>
<p>接着配置模型的超参数配置文件，如classification/configs/net/alexnet.yaml里，可设置batch_size,epoch,lr等超参数</p>
<p>最后，在classification/utils/train.py里修改cfg_data_file变量和cfg_net_file，更改成你需要的配置文件的名称。修改后了之后，直接在终端运行以下命令即可</p>

```shell
python classification/utils/train.py
```


## 作者
<font color="blue">Ricardo</font>
