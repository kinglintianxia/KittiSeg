# KittiSeg note.
---

# FCN论文中作者给出的在VOC2011数据集上的 `Pixel Accuracy`为90.3, mean IoU（即多个类别IoU的平均值，具体看我之前给你写的‘语义分割评价指标‘，IoU即单个类别计算结果，IoU=TP/(TP+FN+FP)）为62.7
# Kitti Road benchmark（http://www.cvlibs.net/datasets/kitti/eval_road.php）
* 目前最好的模型MaxF1:97.05 %， AP：93.53 %； MultiNet 分别为：93.99 % 	93.24 %
* Kitti Road 用MaxF1和AP作为评价指标，这都是像素分类的评价指标，应该时数据集只有单个类别，这样评价比较合理）

# CityScapes benchmark（https://www.cityscapes-dataset.com/benchmarks/#scene-labeling-task）
* 目前最好 Class IoU：83.6，之前跑过的DeepLabv3+为82.1。
* CityScapes 用IoU作为评价指标，这都是语义分割的评价指标，应该是数据集有多个类别，这样评价比较合理

# VOC2012 benchmark（http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php?challengeid=11&compid=6#KEY_FCN-8s）
* 目前最好 DeepLabv3+_JFT	mIoU: 89.0, DeepLabv3+ 为87.8， 	原生FCN-8s（模型结构同KittiSeg）mIoU：62.2

----------------------------------
# 自己做的实验。
* KittiSeg Deconvelution 部分
```python
Shape of Validation/scale5/block3/Relu:0[1 12 39 2048]
Shape of upscore2[1 24 78 2]
Shape of upscore4[1 48 156 2]
Shape of upscore32[1 384 1248 2]
```

# 1. KittiSeg(FCN-8s) 在 Kitti Road 数据集上训练结果(在kitti road评价)（作者给的模型，基于VGG16）：
```shell
2019-01-24 22:55:40,039 INFO Evaluation Succesfull. Results:
2019-01-24 22:55:40,040 INFO     MaxF1  :  96.0821 
2019-01-24 22:55:40,040 INFO     BestThresh  :  14.5098 
2019-01-24 22:55:40,040 INFO     Average Precision  :  92.3620 
2019-01-24 22:55:40,040 INFO     Pixel Accuracy  :  97.8370 
2019-01-24 22:55:40,040 INFO     IOU  :  89.4572 
2019-01-24 22:55:40,040 INFO     Speed (msec)  :  88.4474 
2019-01-24 22:55:40,040 INFO     Speed (fps)  :  11.3062 
```

# 2. KittiSeg(FCN) 在 Kitti Road 数据集上训练结果(在kitti road评价)（基于ResNet101）：
```shell
2019-01-24 23:20:08,677 INFO Evaluation Succesfull. Results:
2019-01-24 23:20:08,677 INFO     [train] MaxF1  :  99.6044 
2019-01-24 23:20:08,677 INFO     [train] BestThresh  :  61.5686 
2019-01-24 23:20:08,678 INFO     [train] Average Precision  :  92.5433 
2019-01-24 23:20:08,678 INFO     [train] Pixel Accuracy  :  99.4363 
2019-01-24 23:20:08,678 INFO     [train] IOU  :  98.3407 
2019-01-24 23:20:08,678 INFO     [val] MaxF1  :  96.6176 
2019-01-24 23:20:08,678 INFO     [val] BestThresh  :  31.7647 
2019-01-24 23:20:08,678 INFO     [val] Average Precision  :  92.0721 
2019-01-24 23:20:08,678 INFO     [val] Pixel Accuracy  :  98.4305 
2019-01-24 23:20:08,678 INFO     [val] IOU  :  92.8428 
2019-01-24 23:20:08,678 INFO     Speed (msec)  :  62.4584 
2019-01-24 23:20:08,678 INFO     Speed (fps)  :  16.0107 
```

# 3. KittiSeg(FCN) 在 Kitti Road + CityScapes 数据集上训练(在kitti road评价)结果（基于ResNet101）：
```shell
2019-01-24 23:24:14,275 INFO Evaluation Succesfull. Results:
2019-01-24 23:24:14,275 INFO     [train] MaxF1  :  98.1683 
2019-01-24 23:24:14,275 INFO     [train] BestThresh  :  76.8627 
2019-01-24 23:24:14,275 INFO     [train] Average Precision  :  92.5292 
2019-01-24 23:24:14,275 INFO     [train] Pixel Accuracy  :  98.7130 
2019-01-24 23:24:14,275 INFO     [train] IOU  :  94.5232 
2019-01-24 23:24:14,275 INFO     [val] MaxF1  :  96.7849 
2019-01-24 23:24:14,276 INFO     [val] BestThresh  :  67.0588 
2019-01-24 23:24:14,276 INFO     [val] Average Precision  :  92.3420 
2019-01-24 23:24:14,276 INFO     [val] Pixel Accuracy  :  98.3321 
2019-01-24 23:24:14,276 INFO     [val] IOU  :  92.4040 
2019-01-24 23:24:14,276 INFO     Speed (msec)  :  49.6830 
2019-01-24 23:24:14,276 INFO     Speed (fps)  :  20.1276 
```

# 3. KittiSeg(FCN) 在 Kitti Road + CityScapes 数据集上训练(在Kitti Road + CityScapes验证集上评价)结果（基于ResNet101）：
```shell
2019-01-25 00:20:39,907 INFO Evaluation Succesfull. Results:
2019-01-25 00:20:39,907 INFO     [val] MaxF1  :  96.3243 
2019-01-25 00:20:39,907 INFO     [val] BestThresh  :  72.9412 
2019-01-25 00:20:39,907 INFO     [val] Average Precision  :  92.6267 
2019-01-25 00:20:39,908 INFO     [val] Pixel Accuracy  :  96.9362 
2019-01-25 00:20:39,908 INFO     [val] IOU  :  91.3910 
2019-01-25 00:20:39,908 INFO     Speed (msec)  :  47.1982 
2019-01-25 00:20:39,908 INFO     Speed (fps)  :  21.1872 
```

# 4. KittiSeg(FCN) 在 Kitti Road + CityScapes 数据集上训练(在kitti road评价)结果（基于ResNet50）：
```python
2019-01-24 23:29:14,570 INFO Evaluation Succesfull. Results:
2019-01-24 23:29:14,570 INFO     [train] MaxF1  :  97.0180 
2019-01-24 23:29:14,570 INFO     [train] BestThresh  :  68.6275 
2019-01-24 23:29:14,570 INFO     [train] Average Precision  :  92.4995 
2019-01-24 23:29:14,570 INFO     [train] Pixel Accuracy  :  98.2073 
2019-01-24 23:29:14,571 INFO     [train] IOU  :  91.9377 
2019-01-24 23:29:14,571 INFO     [val] MaxF1  :  95.2738 
2019-01-24 23:29:14,571 INFO     [val] BestThresh  :  52.1569 
2019-01-24 23:29:14,571 INFO     [val] Average Precision  :  92.1730 
2019-01-24 23:29:14,571 INFO     [val] Pixel Accuracy  :  97.5686 
2019-01-24 23:29:14,571 INFO     [val] IOU  :  88.4408 
2019-01-24 23:29:14,571 INFO     Speed (msec)  :  35.1630 
2019-01-24 23:29:14,571 INFO     Speed (fps)  :  28.4390
```




