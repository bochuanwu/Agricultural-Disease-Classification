# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 11:56:50 2018

@author: 16703
"""
from keras.utils import np_utils
import config
import json
from sklearn.utils import shuffle
import cv2
import random
import numpy as np
from math import ceil
import tensorflow as tf
batch_feature=[]
#初始化sess,或回复保存的sess
def start_or_restore_training(sess,saver,checkpoint_dir):
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print('Restore the model from checkpoint %s' % ckpt.model_checkpoint_path)
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
        step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
    else:
        sess.run(tf.global_variables_initializer())
        step = 1
        print('start training from new state')
    return sess,step

#含中文路径读取图片方法
def cv_imread(filepath):
    img = cv2.imdecode(np.fromfile(filepath,dtype=np.uint8),-1)
    return img

#等比例缩放图片,size为最边短边长
def resize_img(img,size):
    h = img.shape[0]
    w = img.shape[1]
    scale = max(size/h,size/w)
    resized_img = cv2.resize( img, (int(h*scale),int(w*scale)) )
    return resized_img

#对缩放图片进行随机切割,要求输入图片其中一边与切割大小相等
def random_crop(img,size):
    h = img.shape[0]
    w = img.shape[1]
    if h>w and h>size:
        offset = random.randint(0, h - size)
        croped_img = img[offset:offset+size, :]
    elif h<w and w>size:
        offset = random.randint(0, w - size)
        croped_img = img[:, offset:offset+size]
    elif h==w and h>size:
        offset = random.randint(0, h - size)
        croped_img=img[offset:offset+size, offset:offset+size]
    else:croped_img=img
    return croped_img

def load_feature(img_path):
    img = cv_imread(img_path)
    norm_img= img/255.
    crop = resize_img(norm_img,config.INPUT_SIZE)
    crop = random_crop(crop,config.INPUT_SIZE)
    crop = cv2.resize(crop, (config.INPUT_SIZE, config.INPUT_SIZE))
    return crop

def process_annotation(anno_file,dir):
    with open(anno_file) as file:
        annotations = json.load(file)
        img_paths = []
        labels = []
        for anno in annotations:
            img_paths.append(dir + anno["image_id"])
            labels.append(anno["disease_class"])
      
    return img_paths, labels

def make_one_hot(data1):
    return (np.arange(61)==data1[:,None]).astype(np.integer)

def data_generator(img_paths,labels,batch_size,is_shuffle=True):
    if is_shuffle:
        img_paths,labels = shuffle(img_paths,labels)
    num_sample = len(img_paths)
    print(num_sample)
    while True:
        if is_shuffle:
            img_paths, labels = shuffle(img_paths, labels)
            
        for offset in range(0,num_sample,batch_size):
            batch_paths = img_paths[offset:offset+batch_size]
            batch_labels = labels[offset:offset+batch_size]
            batch_labels=np.array(batch_labels)
         
            batch_features = [load_feature(path) for path in batch_paths]
            batch_labels =np_utils.to_categorical(batch_labels,num_classes=61)
           
            batch_feature = np.array(batch_features)
            yield batch_feature, batch_labels

def validation(sess,acc,x,y,rate,anno_file,dir,batch_size):
    img_paths,labels = process_annotation(anno_file,dir)
    data_gen = data_generator(img_paths,labels,batch_size,is_shuffle=False)
    num_sample = len(img_paths)
    num_it = ceil(num_sample/batch_size)

    total_accuracy = 0
   

    for i in range(num_it):
        features,labels = next(data_gen)
        accuracy = sess.run(acc,feed_dict={x:features, y:labels, rate:1.0})
        total_accuracy += accuracy
       

    return total_accuracy/num_it