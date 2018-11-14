# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 21:46:37 2018

@author: 16703
"""



from keras.models import load_model
import os
import util
import config
import numpy as np
import json
import cv2

def load_feature(img_path):
    img = util.cv_imread(img_path)
    crop= img / 255.
    crop = crop - 0.5
    crop = crop * 2.  
    crop = cv2.resize(crop, (config.INPUT_SIZE, config.INPUT_SIZE))
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
#trian_img_paths,tnames = datasettest('./AgriculturalDisease_testA/images/')
nums=[]
train_data_gen = datatest_generator(trian_img_paths,tnames,batch_size)
model = load_model('my_model_invectionV31.h5')
count=0
while 1:

    xtest,tnames = next(train_data_gen)
    if tnames ==   '000f74d036a32b5afc286336e077e26e.jpg' and count>10:
    #if tnames == '0003faa8-4b27-4c65-bf42-6d9e352ca1a5___RS_Late.B 4946.JPG'and count>10:
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


