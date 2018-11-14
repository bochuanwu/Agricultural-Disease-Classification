# -*- coding: utf-8 -*-
import cv2
from keras.utils import np_utils
import numpy as np
from keras import layers
from keras.layers import Input,Dense,Activation,ZeroPadding2D,\
    BatchNormalization,Flatten,Conv2D,AveragePooling2D,MaxPooling2D
import os
import keras
from keras.models import Model
import config
import util
import json
from sklearn.utils import shuffle
import random 
import keras.backend as K
K.set_image_data_format("channels_last")
K.set_learning_phase(1)
from keras.callbacks import ModelCheckpoint


seed = 7
np.random.seed(seed)
from keras.callbacks import ReduceLROnPlateau

from keras.initializers import glorot_uniform

import keras.backend as K
K.set_image_data_format("channels_last")
K.set_learning_phase(1)
 


img_width, img_height = 229, 229

train_data_dir = './train/'
validation_data_dir = './validation/'
nb_train_samples = 32739
nb_validation_samples = 4514
nb_epoch = 20

#恒等模块——identity_block
def identity_block(X,f,filters,stage,block):
    """
    三层的恒等残差块
    param :
    X -- 输入的张量，维度为（m, n_H_prev, n_W_prev, n_C_prev）
    f -- 整数，指定主路径的中间 CONV 窗口的形状
    filters -- python整数列表，定义主路径的CONV层中的过滤器数目
    stage -- 整数，用于命名层，取决于它们在网络中的位置
    block --字符串/字符，用于命名层，取决于它们在网络中的位置
    return:
    X -- 三层的恒等残差块的输出，维度为：(n_H, n_W, n_C)
    """
    #定义基本的名字
    conv_name_base = "res"+str(stage)+block+"_branch"
    bn_name_base = "bn"+str(stage)+block+"_branch"
 
    #过滤器
    F1,F2,F3 = filters
 
    #保存输入值,后面将需要添加回主路径
    X_shortcut = X
 
    #主路径第一部分
    X = Conv2D(filters=F1,kernel_size=(1,1),strides=(1,1),padding="valid",
               name=conv_name_base+"2a",kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3,name=bn_name_base+"2a")(X)
    X = Activation("relu")(X)
 
    # 主路径第二部分
    X = Conv2D(filters=F2,kernel_size=(f,f),strides=(1,1),padding="same",
               name=conv_name_base+"2b",kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3,name=bn_name_base+"2b")(X)
    X = Activation("relu")(X)
 
    # 主路径第三部分
    X = Conv2D(filters=F3,kernel_size=(1,1),strides=(1,1),padding="valid",
               name=conv_name_base+"2c",kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3,name=bn_name_base+"2c")(X)
 
    # 主路径最后部分,为主路径添加shortcut并通过relu激活
    X = layers.add([X,X_shortcut])
    X = Activation("relu")(X)
 
    return X
 
#卷积残差块——convolutional_block
def convolutional_block(X,f,filters,stage,block,s=2):
    """
    param :
    X -- 输入的张量，维度为（m, n_H_prev, n_W_prev, n_C_prev）
    f -- 整数，指定主路径的中间 CONV 窗口的形状（过滤器大小，ResNet中f=3）
    filters -- python整数列表，定义主路径的CONV层中过滤器的数目
    stage -- 整数，用于命名层，取决于它们在网络中的位置
    block --字符串/字符，用于命名层，取决于它们在网络中的位置
    s -- 整数，指定使用的步幅
    return:
    X -- 卷积残差块的输出，维度为：(n_H, n_W, n_C)
    """
    # 定义基本的名字
    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"
 
    # 过滤器
    F1, F2, F3 = filters
 
    # 保存输入值,后面将需要添加回主路径
    X_shortcut = X
 
    # 主路径第一部分
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding="valid",
               name=conv_name_base + "2a", kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + "2a")(X)
    X = Activation("relu")(X)
 
    # 主路径第二部分
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding="same",
               name=conv_name_base + "2b", kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + "2b")(X)
    X = Activation("relu")(X)
 
    # 主路径第三部分
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding="valid",
               name=conv_name_base + "2c", kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + "2c")(X)
 
    #shortcut路径
    X_shortcut = Conv2D(filters=F3,kernel_size=(1,1),strides=(s,s),padding="valid",
               name=conv_name_base+"1",kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3,name=bn_name_base+"1")(X_shortcut)
 
    # 主路径最后部分,为主路径添加shortcut并通过relu激活
    X = layers.add([X, X_shortcut])
    X = Activation("relu")(X)
 
    return X
 
#50层ResNet模型构建
def ResNet50(input_shape = (229,229,3),classes = 61):
    """
    构建50层的ResNet,结构为：
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER
    param :
    input_shape -- 数据集图片的维度
    classes -- 整数，分类的数目
    return:
    model -- Keras中的模型实例
    """
    #将输入定义为维度大小为 input_shape的张量
    X_input = Input(input_shape)
 
    # Zero-Padding
    X = ZeroPadding2D((3,3))(X_input)
 
    # Stage 1
    X = Conv2D(64,kernel_size=(5,5),strides=(2,2),name="conv1",kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3,name="bn_conv1")(X)
    X = Activation("relu")(X)
    X = MaxPooling2D(pool_size=(3,3),strides=(2,2))(X)
 
    # Stage 2
    X = convolutional_block(X,f=3,filters=[64,64,256],stage=2,block="a",s=1)
    X = identity_block(X,f=3,filters=[64,64,256],stage=2,block="b")
    X = identity_block(X,f=3,filters=[64,64,256],stage=2,block="c")
 
    #Stage 3
    X = convolutional_block(X,f=3,filters=[128,128,512],stage=3,block="a",s=2)
    X = identity_block(X,f=3,filters=[128,128,512],stage=3,block="b")
    X = identity_block(X,f=3,filters=[128,128,512],stage=3,block="c")
    X = identity_block(X,f=3,filters=[128,128,512],stage=3,block="d")
 
    # Stage 4
    X = convolutional_block(X,f=3,filters=[256,256,1024],stage=4,block="a",s=2)
    X = identity_block(X,f=3,filters=[256,256,1024],stage=4,block="b")
    X = identity_block(X,f=3,filters=[256,256,1024],stage=4,block="c")
    X = identity_block(X,f=3,filters=[256,256,1024],stage=4,block="d")
    X = identity_block(X,f=3,filters=[256,256,1024],stage=4,block="e")
    X = identity_block(X,f=3,filters=[256,256,1024],stage=4,block="f")
 
    #Stage 5
    X = convolutional_block(X,f=3,filters=[512,512,2048],stage=5,block="a",s=2)
    X = identity_block(X,f=3,filters=[256,256,2048],stage=5,block="b")
    X = identity_block(X,f=3,filters=[256,256,2048],stage=5,block="c")
 
    #最后阶段
    #平均池化
    X = AveragePooling2D(pool_size=(2,2))(X)
 
    #输出层
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc61')(X)
 
    #创建模型
    model = Model(inputs=X_input,outputs=X,name="ResNet50")
 
    return model
 
#运行构建的模型图
model = ResNet50(input_shape=(229,229,3),classes=59)
Adam=keras.optimizers.Adam(lr=0.001)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=2, verbose=1, factor=0.1, min_lr=0.00000001)
model.compile(optimizer=Adam, loss='categorical_crossentropy',metrics=['accuracy'])
def rotate(image, angle=15, scale=0.9):
    w = image.shape[1]
    h = image.shape[0]
    #rotate matrix
    M = cv2.getRotationMatrix2D((w/2,h/2), angle, scale)
    #rotate
    image = cv2.warpAffine(image,M,(w,h))
    return image

def custom_blur_demo(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #锐化
    dst = cv2.filter2D(image, -1, kernel=kernel)
    return dst

def relight(imgsrc, alpha=1, bias=0):
    imgsrc = imgsrc.astype(float)
    imgsrc = imgsrc * alpha + bias
    imgsrc[imgsrc < 0] = 0
    imgsrc[imgsrc > 255] = 255
    imgsrc = imgsrc.astype(np.uint8)
    return imgsrc
#加载数据集
def load_feature(img_path):
    img = util.cv_imread(img_path)
    img= relight(img,random.uniform(0.9, 1.1), random.randint(-10, 10))
    crop= img / 255.
    crop = crop - 0.5
    crop = crop * 2.
    r1=random.uniform(0.9, 1.1)
    r2=random.randint(0, 45)
    crop = rotate(crop,angle=r2, scale=r1)
    flip=np.random.randint(0, 3)
    if flip==1:
        flipimg = cv2.flip(crop, 1)
    if flip==0:
        flipimg = cv2.flip(crop, 0)
    if flip==2:
        flipimg = crop    
    crop = cv2.resize(flipimg, (229, 229))
    return crop

def process_test(anno_file,dir):
    with open(anno_file) as file:
        annotations = json.load(file)
        img_paths = []
        labels = []
        for anno in annotations:
            if anno["disease_class"]==44 or anno["disease_class"]==45:
                continue
            img_paths.append(dir + anno["image_id"])
            if anno["disease_class"]>45:
                anno["disease_class"]=anno["disease_class"]-2
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
            batch_labels=np_utils.to_categorical(batch_labels,num_classes=59)
            batch_features = [load_feature(path) for path in batch_paths]
           
            batch_feature = np.array(batch_features)
            yield batch_feature, batch_labels
'''
def data_generator(img_paths,labels,batch_size,is_shuffle=True):
    if is_shuffle:
        img_paths,labels = shuffle(img_paths,labels)
    num_sample = len(img_paths)
    print(num_sample)
    while True:
        if is_shuffle:
            img_paths, labels, img_paths1, labels1 = train_test_split(img_paths, labels, test_size=0.1, random_state=seed)
        for offset in range(0,num_sample,batch_size):
            batch_paths = img_paths[offset:offset+batch_size]
            batch_labels = labels[offset:offset+batch_size]
            batch_labels=np.array(batch_labels)
            batch_labels=np_utils.to_categorical(batch_labels,num_classes=61)
            batch_features = [load_feature(path) for path in batch_paths]
           
            batch_feature = np.array(batch_features)
            
            batch_paths1 = img_paths1[offset:offset+batch_size]
            batch_labels1 = labels1[offset:offset+batch_size]
            batch_labels1=np.array(batch_labels1)
            batch_labels1=np_utils.to_categorical(batch_labels1,num_classes=61)
            batch_features1 = [load_feature(path) for path in batch_paths1]
           
            batch_feature = np.array(batch_features)
            yield batch_feature, batch_labels,batch_features1, batch_labels1
            '''

            
batch_size=20
#trian_img_paths,train_labels,trian_img_paths1,train_labels1 = process_test(config.TRAIN_ANNOTATION_FILE,config.TRAIN_DIR)
trian_img_paths,train_labels = process_test(config.TRAIN_ANNOTATION_FILE,config.TRAIN_DIR)
trian_img_paths1,train_labels1 = process_test(config.VAL_ANNOTATION_FILE,config.VAL_DIR)
model.load_weights('weights.hdf5')
checkpoint = ModelCheckpoint('weights.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max', period=1)
#训练模型
model.fit_generator(data_generator(trian_img_paths,train_labels,batch_size),samples_per_epoch=len(train_labels)//32,nb_epoch=20,validation_data=data_generator(trian_img_paths1,train_labels1,batch_size),nb_val_samples=4982//20,callbacks=[checkpoint,learning_rate_reduction])
#model.fit_generator(data_generator(trian_img_paths,train_labels,batch_size),samples_per_epoch=len(train_labels)//32,nb_epoch=10,validation_data=data_generator(trian_img_paths1,train_labels1,batch_size),nb_val_samples=len(train_labels1)//32,callbacks=[checkpoint])
model.save(os.path.join('./', 'my_model_resnet.h5'))

#def my_load_model(resultpath):

    # test data
 #   X = np.array(np.arange(86400)).reshape(2, 120, 120, 3)
  #  Y = [0, 1]
   # X = X.astype('float32')
    #Y = np_utils.to_categorical(Y, 4)

    # the second way : load model structure and weights
   # model = model_from_json(open(os.path.join(resultpath, 'my_model_structure.json')).read())
    #model.load_weights(os.path.join(resultpath, 'my_model_weights.hd5'))
    #model.compile(loss=categorical_crossentropy,
     #             optimizer=Adam(), metrics=['accuracy']) 

   # test_loss, test_acc = model.evaluate(X, Y, verbose=0)
    #print('Test loss:', test_loss)
   # print('Test accuracy:', test_acc)

    #y = model.predict_classes(X)
    #print("predicct is: ", y)

