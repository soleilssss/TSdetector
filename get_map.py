import os
import xml.etree.ElementTree as ET

from PIL import Image
from tqdm import tqdm
import numpy as np
from utils.utils import get_classes
from utils.utils_map import get_coco_map, get_map
from yolo import YOLO

if __name__ == "__main__":
    '''
    Recall和Precision不像AP是一个面积的概念，因此在门限值（Confidence）不同时，网络的Recall和Precision值是不同的。
    默认情况下，本代码计算的Recall和Precision代表的是当门限值（Confidence）为0.5时，所对应的Recall和Precision值。

    受到mAP计算原理的限制，网络在计算mAP时需要获得近乎所有的预测框，这样才可以计算不同门限条件下的Recall和Precision值
    因此，本代码获得的map_out/detection-results/里面的txt的框的数量一般会比直接predict多一些，目的是列出所有可能的预测框，
    '''
    #------------------------------------------------------------------------------------------------------------------#
    #   map_mode用于指定该文件运行时计算的内容
    #   map_mode为0代表整个map计算流程，包括获得预测结果、获得真实框、计算VOC_map。
    #   map_mode为1代表仅仅获得预测结果。
    #   map_mode为2代表仅仅获得真实框。
    #   map_mode为3代表仅仅计算VOC_map。
    #   map_mode为4代表利用COCO工具箱计算当前数据集的0.50:0.95map。需要获得预测结果、获得真实框后并安装pycocotools才行
    #-------------------------------------------------------------------------------------------------------------------#
    map_mode        = 0
    #--------------------------------------------------------------------------------------#
    #   此处的classes_path用于指定需要测量VOC_map的类别
    #   一般情况下与训练和预测所用的classes_path一致即可
    #--------------------------------------------------------------------------------------#
    classes_path    = r'D:\肠镜课题\肠镜中找到的公开数据集\sun_dataset\YOLOX\yolox-timeimage-sim-Lstm-tada\model_data\sun_classes.txt'
    #--------------------------------------------------------------------------------------#
    #   MINOVERLAP用于指定想要获得的mAP0.x，mAP0.x的意义是什么请同学们百度一下。
    #   比如计算mAP0.75，可以设定MINOVERLAP = 0.75。
    #
    #   当某一预测框与真实框重合度大于MINOVERLAP时，该预测框被认为是正样本，否则为负样本。
    #   因此MINOVERLAP的值越大，预测框要预测的越准确才能被认为是正样本，此时算出来的mAP值越低，
    #--------------------------------------------------------------------------------------#
    MINOVERLAP      = 0.5
    #--------------------------------------------------------------------------------------#
    #   受到mAP计算原理的限制，网络在计算mAP时需要获得近乎所有的预测框，这样才可以计算mAP
    #   因此，confidence的值应当设置的尽量小进而获得全部可能的预测框。
    #   
    #   该值一般不调整。因为计算mAP需要获得近乎所有的预测框，此处的confidence不能随便更改。
    #   想要获得不同门限值下的Recall和Precision值，请修改下方的score_threhold。
    #--------------------------------------------------------------------------------------#
    confidence      = 0.001
    #--------------------------------------------------------------------------------------#
    #   预测时使用到的非极大抑制值的大小，越大表示非极大抑制越不严格。
    #   
    #   该值一般不调整。
    #--------------------------------------------------------------------------------------#
    nms_iou         = 0.5
    #---------------------------------------------------------------------------------------------------------------#
    #   Recall和Precision不像AP是一个面积的概念，因此在门限值不同时，网络的Recall和Precision值是不同的。
    #   
    #   默认情况下，本代码计算的Recall和Precision代表的是当门限值为0.5（此处定义为score_threhold）时所对应的Recall和Precision值。
    #   因为计算mAP需要获得近乎所有的预测框，上面定义的confidence不能随便更改。
    #   这里专门定义一个score_threhold用于代表门限值，进而在计算mAP时找到门限值对应的Recall和Precision值。
    #---------------------------------------------------------------------------------------------------------------#
    score_threhold  = 0.5
    #-------------------------------------------------------#
    #   map_vis用于指定是否开启VOC_map计算的可视化
    #-------------------------------------------------------#
    map_vis         = False
    #-------------------------------------------------------#
    #   指向VOC数据集所在的文件夹
    #   默认指向根目录下的VOC数据集
    #-------------------------------------------------------#
    VOCdevkit_path  = r'D:\肠镜课题\肠镜中找到的公开数据集\sun_dataset\data\SUN-Positive'
    #-------------------------------------------------------#
    #   结果输出的文件夹，默认为map_out
    #-------------------------------------------------------#
    map_out_path    = 'sun_out_x_sgd_time_image'
    framerange      = 50
    videorange      = 3

    image_ids = open(os.path.join(VOCdevkit_path, "ImageSets/test.txt")).read().strip().split()

    if not os.path.exists(map_out_path):
        os.makedirs(map_out_path)
    if not os.path.exists(os.path.join(map_out_path, 'ground-truth')):
        os.makedirs(os.path.join(map_out_path, 'ground-truth'))
    if not os.path.exists(os.path.join(map_out_path, 'detection-results')):
        os.makedirs(os.path.join(map_out_path, 'detection-results'))
    if not os.path.exists(os.path.join(map_out_path, 'images-optional')):
        os.makedirs(os.path.join(map_out_path, 'images-optional'))

    class_names, _ = get_classes(classes_path)

    if map_mode == 0 or map_mode == 1:
        print("Load model.")
        yolo = YOLO(confidence = confidence, nms_iou = nms_iou)
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path  = os.path.join(VOCdevkit_path, "JPEGImages/"+image_id)
            jpg_list  =  os.listdir(image_path)
            jpg_list.sort()
            for jpg_id in jpg_list:
                jpg_path = os.path.join(image_path, '%s'%(jpg_id) )
                image    = Image.open(jpg_path)

                search_case = jpg_id.split("image")[0]  #获得这一帧所在的case
                search_pre_frame = [row for row in jpg_list if search_case in row]  #根据所在case找到所有的帧作为前面帧的搜索范围
                search_frame_name_index = search_pre_frame.index(jpg_id)  #找到这一帧在搜索范围的位置
                #获取前面几帧的索引
                #------------------------------#
                #   读取图像的preimage图像
                #------------------------------#
                # search_frame_name_index = jpg_list.index(jpg_id)  #找到这一帧在搜索范围的位置
                if search_frame_name_index == 0:
                    search_frame_name_index = 1
                else:
                    search_frame_name_index = search_frame_name_index

                left = max(search_frame_name_index - framerange, 0)
                right = min(search_frame_name_index + framerange, len(jpg_list)-1) + 1
                search_range = jpg_list[left:search_frame_name_index]  #搜索就在frame_range左右的范围内作为search_frame
                listname=np.arange(0,len(search_range))  #选取0-search_frame之间的所有帧
        # 总的来说就是在listname里面抽取videorange数量个帧
                if len(listname)>=1.5*videorange:  #如果大于1.5倍的videorange（3），就继续在listname里面只抽50帧
                    serrange=np.random.choice(listname,videorange,replace=False)
                    serrange.sort()
                    if serrange[-1]!=listname[-1]:
                        serrange[-1]=listname[-1]
                elif len(listname)!=0:
                    serrange=np.random.choice(listname,videorange,replace=True)
                    serrange.sort()
                    if serrange[-1]!=listname[-1]:
                        serrange[-1]=listname[-1]
                else:
                    serrange=np.zeros((videorange))
                #得到template和search_frame的图片路径和gt    
                previous_path=[]
                #得到search_frame附近的videorange帧的图片路径和gt 
                for item in serrange:
                    previous_path.append(os.path.join(image_path, search_range[int(item)]))
                ###########################################
                if map_vis:
                    image.save(os.path.join(map_out_path, "images-optional/%s_%s" %(image_id, jpg_id)))
                yolo.get_map_txt(image_id, jpg_id, image,previous_path, class_names, map_out_path,videorange)
                # r_image = yolo.detect_image(image, crop = False, count=False)
                # r_image.show()
        print("Get predict result done.")
        
    if map_mode == 0 or map_mode == 2:
        print("Get ground truth result.")
        for image_id in tqdm(image_ids):
            image_path  = os.path.join(VOCdevkit_path, "JPEGImages/"+image_id)
            jpg_list  =  os.listdir(image_path)
            case_txt = open(VOCdevkit_path + '/Annotations/%s.txt'%(image_id)).readlines()
            for jpg_id in jpg_list:
                gt_out_path = os.path.join(map_out_path, "ground-truth")
                with open(gt_out_path + "/%s_%s.txt"%(image_id, jpg_id.split('.')[0]),"w") as new_f:
                    index = [i for i,x in enumerate(case_txt) if (jpg_id in x)]
                    text = case_txt[index[0]].split()[1].split(',')
                    new_f.write("%s %s %s %s %s\n" % ('polyp', text[0], text[1], text[2], text[3]))
        print("Get ground truth result done.")

    if map_mode == 0 or map_mode == 3:
        print("Get map.")
        get_map(MINOVERLAP, True, score_threhold = score_threhold, path = map_out_path)
        print("Get map done.")

    if map_mode == 0 or map_mode == 4:
        print("Get map.")
        get_coco_map(class_names = class_names, path = map_out_path)
        print("Get map done.")
