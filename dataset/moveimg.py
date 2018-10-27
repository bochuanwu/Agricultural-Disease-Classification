          # -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 14:13:09 2018

@author: 16703
"""


import os
import shutil 
import json

#建立图片名与类别相

def process_annotation(anno_file,dir1):
    with open(anno_file) as file:
        annotations = json.load(file)
        img_paths = []
        labels = []
        for anno in annotations:
            img_paths.append(dir1 + anno["image_id"])
            labels.append(anno["disease_class"])
      
    return img_paths, labels
img_paths, labels=process_annotation('./AgriculturalDisease_train_annotations1.json','./images/')
#创建文件夹
for i in range(61):
    os.mkdir(str(i))
#进行分类
for i in range(0,61):    
    for l in range(len(labels)):
        if labels[l]==i:
            img=img_paths[l]
            j=str(i)    
            shutil.copy(img,j)
