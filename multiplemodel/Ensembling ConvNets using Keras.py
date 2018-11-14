# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 17:17:32 2018

@author: 16703
"""

from keras.models import load_model
import os
#from keras.applications.densenet import DenseNet121
#from keras.applications.resnet import ResNet50
#from keras.applications.xception import Xception
from keras.layers import Input,Dense,GlobalMaxPooling2D,Dropout
from keras.models import Model
from keras.layers import average
from keras.layers import maximum
#import keras
import util
#import random
import config
import numpy as np
from sklearn.utils import shuffle
import cv2
import json


input_layer=Input(shape=(250,250,3))

models_output=[]
model1 = load_model('my_model_invectionV3_87.h5')
model2 = load_model('my_model_multiplemodel_874.h5')
model3 = load_model('my_model_xception.h5')

models = [model1, model2, model3]
for i, m in enumerate(models):
        # Keras needs all the models to be named differently
    m.name = 'model_' + str(i)
    models_output.append(m(input_layer))
out = maximum(models_output)
        # Build model from same input and outputs
model = Model(inputs=input_layer, outputs=out)




def load_feature(img_path):
    img = util.cv_imread(img_path)
    crop= img / 255.
    crop = crop - 0.5
    crop = crop * 2.  
    crop = cv2.resize(crop, (250, 250))
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
#trian_img_paths,tnames = datasettest(config.VAL_DIR)
trian_img_paths,tnames = datasettest('./AgriculturalDisease_testB/images/')
nums=[]
train_data_gen = datatest_generator(trian_img_paths,tnames,batch_size)
count=0
while 1:

    xtest,tnames = next(train_data_gen)
   # if tnames ==   '000f74d036a32b5afc286336e077e26e.jpg' and count>10:
    if count==4513:
        jsonData = json.dumps(nums)
       # fileObject = open('invectionV3test.json', 'w')
        fileObject = open('ensemblingq.json', 'w')
        fileObject.write(jsonData)
        fileObject.close()
    
        break
    count+=1
    out = model.predict(xtest)
    out=  np.argmax(out, 1) 

    if int(out) > 43:
        out=out+2
    num={"image_id":str(tnames[0]),"disease_class":int(out)}
    print(num)
    print(count)
    nums.append(num)