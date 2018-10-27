# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 14:31:09 2018

@author: 16703
"""
import cv2
from keras.utils import np_utils
import numpy as np
from keras import layers
from keras.layers import Input,Dense,Activation,ZeroPadding2D,\
    BatchNormalization,Flatten,Conv2D,AveragePooling2D,MaxPooling2D
import os
from keras.initializers import glorot_uniform
from keras.models import Model
import config
import util
import json
from sklearn.utils import shuffle
import keras
import keras.backend as K
import random
import tensorflow as tf
K.set_image_data_format("channels_last")
K.set_learning_phase(1)
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

from keras.initializers import glorot_normal
name_initializer="glorot_normal"
base_model=keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet', input_tensor=None, input_shape=(224,224,3), pooling='avg')

x = base_model.output
x = Dense(2048, activation='relu', name='fc1')(x)
predictions = Dense(61, activation='softmax', name='predictions')(x)
model = Model(inputs=base_model.input, outputs=predictions)
Adam=keras.optimizers.Adam(lr=0.0001)
model.compile(optimizer=Adam, loss='categorical_crossentropy',metrics=['accuracy'])


def relight(imgsrc, alpha=1, bias=0):
    imgsrc = imgsrc.astype(float)
    imgsrc = imgsrc * alpha + bias
    imgsrc[imgsrc < 0] = 0
    imgsrc[imgsrc > 255] = 255
    imgsrc = imgsrc.astype(np.uint8)
    return imgsrc


def random_crop(img,size):
    h = 200
    w = 200
    # scale the image
    scale = np.random.uniform() / 10. + 1.
    img = cv2.resize(img, (0,0), fx = scale, fy = scale)
    max_offx = (scale-1.) * w
    max_offy = (scale-1.) * h
    offx = int(np.random.uniform() * max_offx)
    offy = int(np.random.uniform() * max_offy)
    img = img[offy : (offy + h), offx : (offx + w)]
        # re-color
    img= relight(img,random.uniform(0.7, 1.2), random.randint(-15, 15))
    img = img/255. 
    img = cv2.resize(img, (config.INPUT_SIZE,config.INPUT_SIZE))

    return img

def load_feature(img_path):
    img = util.cv_imread(img_path)
   
    crop=random_crop(img,224)
    flip=np.random.randint(0, 3)
    if flip==1:
        flipimg = cv2.flip(crop, 1)
    if flip==0:
        flipimg = cv2.flip(crop, 0)
    if flip==2:
        flipimg = crop
    return flipimg

def process_test(anno_file,dir):
    with open(anno_file) as file:
        annotations = json.load(file)
        img_paths = []
        labels = []
        for anno in annotations:
            img_paths.append(dir + anno["image_id"])
            labels.append(anno["disease_class"])
    return img_paths, labels

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
            batch_labels=np_utils.to_categorical(batch_labels,num_classes=61)
            batch_features = [load_feature(path) for path in batch_paths]
           
            batch_feature = np.array(batch_features)
            yield batch_feature, batch_labels
            
batch_size=32
trian_img_paths,train_labels = process_test(config.TRAIN_ANNOTATION_FILE,config.TRAIN_DIR)

trian_img_paths1,train_labels1 = process_test(config.VAL_ANNOTATION_FILE,config.VAL_DIR)

#训练模型

checkpoint = ModelCheckpoint('weights.hdf5', monitor='loss', verbose=1, save_best_only=True, mode='min', period=1)
model.fit_generator(data_generator(trian_img_paths,train_labels,batch_size),samples_per_epoch=32739//32,nb_epoch=50,validation_data=data_generator(trian_img_paths1,train_labels1,batch_size),nb_val_samples=4982//32,callbacks=[checkpoint])

model.save(os.path.join('./', 'my_model_invectionV3.h5'))
