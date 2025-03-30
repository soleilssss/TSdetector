# TSdetector

TSdetector: Temporal–Spatial self-correction collaborative learning for colonoscopy video detection

Our paper has been accepted by Medical Image Analysis 2025.

## Prerequisites
 
requirements.txt

推荐使用torch1.7.1以上的版本。

## Training 

1、数据集的准备
本文使用VOC格式进行训练，训练前需要自己制作好数据集，

2、开始网络训练
训练的参数较多，均在train.py中，大家可以在下载库后仔细看注释，其中最重要的部分依然是train.py里的classes_path。
修改完classes_path后就可以运行train.py开始训练了，在训练多个epoch后，权值会生成在logs文件夹中。

3、训练结果预测
训练结果预测需要用到两个文件，分别是yolo.py和predict.py。在yolo.py里面修改model_path以及classes_path。
model_path指向训练好的权值文件，在logs文件夹里。
classes_path指向检测类别所对应的txt。
完成修改后就可以运行predict.py进行检测了。运行后输入图片路径即可检测。

## Citation

If you find the code useful for your research, please cite our paper.

Wang, Kai-Ni, et al. "TSdetector: Temporal–Spatial self-correction collaborative learning for colonoscopy video detection." Medical Image Analysis 100 (2025): 103384.

