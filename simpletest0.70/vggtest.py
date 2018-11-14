# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 09:53:39 2018

@author: 16703
"""

import json
import util
import config
import cv2
import os
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D,BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense

# path to the model weights files.


# dimensions of our images.
img_width, img_height = 224, 224

train_data_dir = './train'
validation_data_dir = './validation'
nb_train_samples = 32739
nb_validation_samples = 4982
nb_epoch = 50

# build the VGG16 network
model = Sequential()
model.add(ZeroPadding2D((1, 1), input_shape=( img_width, img_height,3)))

model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))

model.add(BatchNormalization())
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
model.add(BatchNormalization())
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
model.add(BatchNormalization())
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
model.add(BatchNormalization())
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
model.add(BatchNormalization())
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
model.add(BatchNormalization())
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
model.add(BatchNormalization())
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
model.add(BatchNormalization())
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

# load the weights of the VGG16 networks
# (trained on ImageNet, won the ILSVRC competition in 2014)
# note: when there is a complete match between your model definition
# and your weight savefile, you can simply call model.load_weights(filename)


# build a classifier model to put on top of the convolutional model
top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(61, activation='softmax'))

# note that it is necessary to start with a fully-trained
# classifier, including the top classifier,
# in order to successfully do fine-tuning


# note that it is necessary to start with a fully-trained
# classifier, including the top classifier,
# in order to successfully do fine-tuning


# add the model on top of the convolutional base
model.add(top_model)
model.load_weights('vgg_weights.h5')


def load_feature(img_path):
    img = util.cv_imread(img_path)
    norm_img= img*(1./255)
    
    crop = cv2.resize(norm_img, (config.INPUT_SIZE, config.INPUT_SIZE))
    return crop

def datasettest(paths):
    testimgs=[]
    testnames=[]
    for filename in os.listdir(paths):
        file=paths+'/'+filename
        testimgs.append(file)
        testnames.append(filename)
    return testimgs,testnames

def datatest_generator(img_paths,name,batch_size):
    num_sample = len(img_paths)
    print(num_sample)
    while True:
        for offset in range(0,num_sample,batch_size):
            batch_paths = img_paths[offset:offset+batch_size]
            batch_labels = name[offset:offset+batch_size]
            batch_labels=np.array(batch_labels)
            
            batch_features = [load_feature(path) for path in batch_paths]
           
            batch_feature = np.array(batch_features)
            yield batch_feature, batch_labels

batch_size=1
#trian_img_paths,train_labels = util.process_annotation(config.TRAIN_ANNOTATION_FILE,config.TRAIN_DIR)
#train_data_gen = util.data_generator(trian_img_paths,train_labels,batch_size)
imgs,names =datasettest(config.VAL_DIR)
#num_batch = len(names) // batch_size
train_data_gen = datatest_generator(imgs,names,batch_size)
nums=[]
while 1:
    x,name=next(train_data_gen)
# 【5】测试数据
    preds = model.predict(x)
    predict = np.argmax(preds, 1) 
    print(predict)
    num={"image_id":str(name[0]),"disease_class":int(predict)}
    #num={"image_id":name[0],"disease_class":int(predict)}
    nums.append(num)
    #count+=1
    #print(count)
    
 # 在多标签分类中，大多使用binary_crossentropy损失而不是通常在多类分类中使用的categorical_crossentropy损失函数
    #if name == 'u=3337154909,2169261495&fm=27&gp=0.jpg': 
    if name[0] ==   'u=2520622969,1450176384&fm=200&gp=0.jpg':#VAL
        jsonData = json.dumps(nums)
        fileObject = open('json_file.json', 'w')
        fileObject.write(jsonData)
        fileObject.close()
        print('save done')
        break