# PortraitNet-Paddle

## PortraitNet

本项目基于paddlepaddle框架复现PortraitNet，并参加百度第三届论文复现赛，将在2021年5月15日比赛完后提供AIStudio链接～敬请期待

参考项目：

[dong-x16-PortraitNet](https://github.com/dong-x16/PortraitNet)

## 数据与环境准备
PaddleVersion: `paddlepaddle-gpu       2.0.2`  
CocoAPI: `pip install git+https://gitee.com/GuoQuanhao/cocoapi.git#subdirectory=PythonAPI`，我在Data目录下下载了`Coco`源码，注意到这并不是必须的
**有关如何在Paddle环境配置Coco，可参见[【飞桨】Win10 环境安装 Paddledetection](https://blog.csdn.net/qq_39567427/article/details/109029798)，里面包含了配置Coco的多种方式

***
- [EG1800](https://pan.baidu.com/s/1myEBdEmGz6ufniU3i1e6Uw) Since several image URL links are invalid in the [original EG1800 dataset](http://xiaoyongshen.me/webpage_portrait/index.html), we finally use 1447 images for training and 289 images for validation. 

- [Supervise-Portrait](https://pan.baidu.com/s/1uBtCsLj156e_iy3DtkvjQQ) Supervise-Portrait is a portrait segmentation dataset collected from the public human segmentation dataset [Supervise.ly](https://supervise.ly/) using the same data process as EG1800.

**根据链接下载数据集，本项目已挂载相应数据集**

准备完成后，你应该具有以下初始目录结构
```
/home/aistudio
|───Data
└───────EG1800
└───────Supervisely_face
└───────cocoapi(optional)
└───PortraitNet
└───────config
└───────myTest
└───────myTrain
└───────data
└───────model
└───────util
```
**在`config`文件夹下包含了训练的配置文档，注意根据如果在本地部署你需要修改里面的数据文件路径**

## 训练
训练的参数配置在`config`文件夹下，里面有四种配置文件（其中有一个是空的😂）  
你需要进入`myTrain文件夹`
```
cd /home/aistudio/PortraitNet/myTrain
python train.py
```
`training.log`包含部分训练过程中部分打印输出
训练开始后你会在`/home/aistudio/PortraitNet/myexp`下看到保存的模型文件以及训练日志
```
/home/aistudio/PortraitNet/myexp/mobilenetv2_eg1800/single_224_without_group
|───log
└───────train
└───────────vdlrecords.xxxx.log
└───────test
└───────────vdlrecords.xxxx.log
└───model_best.pdparams
└───optimizer_best.pdopt
└───checkpoint.pdparams
└───model_best.pdparams
└───checkpoint.pdopt
```
通过AIStudio可以方便的启动VisualDL，本地部署可以采取`visualdl --logdir XXX`的方式启动

<img src="https://ai-studio-static-online.cdn.bcebos.com/fe3026d332dd49da85780ae3d589e1a4457836431a504021ba505d1d3fb9ed5a" width="500"/>
<img src="https://ai-studio-static-online.cdn.bcebos.com/f56cc05ffca041f0818ec73096a2b343427cf171af3d4a9ab6253b1af003ffab" width="500"/>

**训练的日志文件可通过`wget ftp://207.246.98.85/myexp.zip`命令从我的服务器上获取**

<img src="https://ai-studio-static-online.cdn.bcebos.com/672fc80308c04b55974e03c7d41c32a03f0be1b160444b9686b040a41d1f73b0" width="500"/>

**我已经在`myTrain`文件夹下提供了`paddle`版本的预训练模型，这来源于原作者提供的`torch`模型**  
下面列出了每个权重文件对应的模型配置  
`mobilenetv2_eg1800_with_two_auxiliary.pdparams`
```python
import model_mobilenetv2_seg_small as modellib

netmodel = modellib.MobileNetV2(n_class=2, 
        useUpsample=False, 
        useDeconvGroup=False, 
        addEdge=True, 
        channelRatio=1.0, 
        minChannel=16, 
        weightInit=True,
        video=False)
```

`model_mobilenetv2_with_prior_channel`
```python
import model_mobilenetv2_seg_small as modellib

netmodel = modellib.MobileNetV2(n_class=2, 
        useUpsample=False, 
        useDeconvGroup=False, 
        addEdge=False, 
        channelRatio=1.0, 
        minChannel=16, 
        weightInit=True,
        video=True)
```
`mobilenetv2_supervise_portrait_with_two_auxiliary_losses`
```python
import model_mobilenetv2_seg_small as modellib

netmodel = modellib.MobileNetV2(n_class=2, 
        useUpsample=False, 
        useDeconvGroup=False, 
        addEdge=True, 
        channelRatio=1.0, 
        minChannel=16, 
        weightInit=True,
        video=False)
```
***注意到：如果模型结构与权重文件不适配将会导致加载权重出错***

## 评估与测试

[模型下载](https://pan.baidu.com/s/124OdgrTyZo2nxfSl6Gn0BQ)

提取码：8nno

[AIStudio链接](https://aistudio.baidu.com/aistudio/projectdetail/1885915?channel=0&channelType=0&shared=1)

### EvalModel
评估代码`EvalModel.py`放置在`/home/aistudio/PortraitNet/myTest`文件夹下，同样你需要修改`EvalModel.py`中的部分目录路径来评估自己的模型，本项目中采用`mobilenetv2_eg1800_with_two_auxiliary_losses.pdparams`模型文件评估
```
cd /home/aistudio/PortraitNet/myTest
python EvalModel.py
```
将得到以下结果
```
289
mean iou:  0.9661549598679801
```

### TestModel
作者提供了一个`douyu_origin.mp4`视频文件用于测试，`/home/aistudio/PortraitNet/myTest`文件夹下提供的`VideoTest.py`用于测试，需要注意的是(aistudio不显示后缀为`ipynb`的文件，所以我使用`.py`文件保存)，你需要将`VideoTest.py`中的内容复制`ipynb`文件中，这样方便可视化，我已使用`######...`将代码分成三段，最终你会得到如下结果  


<img src="https://ai-studio-static-online.cdn.bcebos.com/0040c8273d2c432d9a2e740fcb7ef9bdf2cb8bef016a4437ab4fb446bbd1eb8e" width="300"/>

```
cnt:  100
cnt:  200
cnt:  300
cnt:  400
cnt:  500
cnt:  600
finish
```

并得到`result.mp4`媒体文件，如下所示

<img src="https://img-blog.csdnimg.cn/20210501174355753.gif" width="500"/>

# **关于作者**
<img src="https://ai-studio-static-online.cdn.bcebos.com/cb9a1e29b78b43699f04bde668d4fc534aa68085ba324f3fbcb414f099b5a042" width="100"/>


| 姓名        |  郭权浩                           |
| --------     | -------- | 
| 学校        | 电子科技大学研2020级     | 
| 研究方向     | 计算机视觉             | 
| 主页        | [Deep Hao的主页](https://blog.csdn.net/qq_39567427?spm=1000.2115.3001.5343) |
如有错误，请及时留言纠正，非常蟹蟹！
后续会有更多论文复现系列推出，欢迎大家有问题留言交流学习，共同进步成长！
