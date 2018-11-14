# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 09:50:05 2018

@author: 16703
"""


import os
from keras.applications.xception import Xception
from keras.layers import Input,Dense,GlobalMaxPooling2D,Dropout
from keras.models import Model
import keras
import util
import config
import numpy as np
import cv2
import json


def Multimodel(cnn_weights_path=None,all_weights_path=None,class_num=61):
    input_layer=Input(shape=(224,224,3))

    xception=Xception(include_top=False,weights=None,input_tensor=input_layer,input_shape=(224,224,3))
    if cnn_weights_path!=None:
        xception.load_weights(cnn_weights_path)

	#对dense_121和xception进行全局最大池化
    top1_model=GlobalMaxPooling2D(input_shape=(7,7,1024),data_format='channels_last')(xception.output)

	#第一个全连接层
    top_model=Dense(units=512,activation="relu")(top1_model)
    top_model=Dropout(rate=0.5)(top_model)
    top_model=Dense(units=class_num,activation="softmax")(top_model)
	
    model=Model(inputs=input_layer,outputs=top_model)
 
	#加载全部的参数
    if all_weights_path:
        model.load_weights(all_weights_path)
    return model
weight_path='./xception_weights_tf_dim_ordering_tf_kernels_notop.h5'
Adam=keras.optimizers.Adam(lr=0.001)
model=Multimodel(cnn_weights_path=weight_path,all_weights_path=None,class_num=59)

model.compile(optimizer=Adam, loss='categorical_crossentropy',metrics=['accuracy'])


def custom_blur_demo(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #锐化
    dst = cv2.filter2D(image, -1, kernel=kernel)
    return dst
def load_feature(img_path):
    img = util.cv_imread(img_path)
    
    crop= img / 255.
    crop = crop - 0.5
    crop = crop * 2.  
    crop = cv2.resize(crop, (224, 224))
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
            batch_names= name[offset:offset+batch_size]
            batch_features = [load_feature(path) for path in batch_paths]
            batch_feature = np.array(batch_features)
            yield batch_feature, np.array(batch_names)

batch_size=1
trian_img_paths,tnames = datasettest(config.VAL_DIR)
#trian_img_paths,tnames = datasettest('./AgriculturalDisease_testB/images/')
nums=[]
train_data_gen = datatest_generator(trian_img_paths,tnames,batch_size)
model.load_weights('weightsxception883.hdf5')
count=0
while 1:

    xtest,tnames = next(train_data_gen)
    if tnames ==   '000f74d036a32b5afc286336e077e26e.jpg' and count>10:
    #if count==4513:
        jsonData = json.dumps(nums)
       # fileObject = open('invectionV3test.json', 'w')
        fileObject = open('invectionV3.json', 'w')
        fileObject.write(jsonData)
        fileObject.close()
    
        break
    count+=1
    out = model.predict(xtest)
    out=  np.argmax(out, 1) 
    if int(out)>=44:
        out=int(out)+2
    num={"image_id":str(tnames[0]),"disease_class":int(out)}
    print(num)
    nums.append(num)
