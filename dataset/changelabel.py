# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 14:50:06 2018

@author: 16703
"""



import json


def process_annotation(anno_file):
    with open(anno_file) as file:
        annotations = json.load(file)
        img_paths = []
        labels = []
        for anno in annotations:
            img_paths.append(anno["image_id"])
            labels.append(anno["disease_class"])
      
    return img_paths, labels
img_paths, labels=process_annotation('./AgriculturalDisease_train_annotations.json')
ls={}
for i,label in enumerate(labels):
    print(label)
    if label <6:
        ls[i]=0#苹果
    elif 6<=label<9:
        ls[i]=1#樱桃
    elif 9<=label<17:
        ls[i]=2#玉米
    elif 17<=label<24:
        ls[i]=3#葡萄
    elif 24<=label<27:
        ls[i] =4#柑桔
    elif 27<=label<30:
        ls[i] = 5#桃
    elif 30<=label<33:
        ls[i] = 6#辣椒
    elif 33<=label<38:
        ls[i] = 7#马铃薯
    elif 38<=label<41:
        ls[i] = 8#草莓
    elif 41<=label<61:
        ls[i]= 9 #番茄
nums=[]
for j,img in enumerate(img_paths):
    num={"image_id":img,"disease_class":ls[j]}
    nums.append(num)
jsonData = json.dumps(nums)
fileObject = open('label_file.json', 'w')
fileObject.write(jsonData)
fileObject.close()
print('save done')
with open('label_file.json', 'r') as file_obj:
            '''写入json文件'''
            n=json.load(file_obj)
            print(n)
