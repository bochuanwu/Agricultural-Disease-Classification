# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 14:09:12 2018

@author: 16703
"""
import config
import util
from keras.utils import np_utils
import json
from sklearn.utils import shuffle
import os
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D,BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense

from keras.optimizers import SGD

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
top_model.add(Dense(1024, activation='relu'))
top_model.add(Dense(1024, activation='relu'))
top_model.add(Dense(61, activation='softmax'))

# note that it is necessary to start with a fully-trained
# classifier, including the top classifier,
# in order to successfully do fine-tuning


# add the model on top of the convolutional base
model.add(top_model)
for layer in model.layers[:25]:
    layer.trainable = True
#model.load_weights('vgg_weights.h5')
# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
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
            batch_features = [util.load_feature(path) for path in batch_paths]
           
            batch_feature = np.array(batch_features)
            yield batch_feature, batch_labels
#训练模型
batch_size=21
trian_img_paths,train_labels = process_test(config.TRAIN_ANNOTATION_FILE,config.TRAIN_DIR)

trian_img_paths1,train_labels1 = process_test(config.VAL_ANNOTATION_FILE,config.VAL_DIR)

#训练模型
model.fit_generator(data_generator(trian_img_paths,train_labels,batch_size),samples_per_epoch=32739//21,nb_epoch=30,validation_data=data_generator(trian_img_paths1,train_labels1,batch_size),nb_val_samples=4982//21)

model.save(os.path.join('./', 'my_model_vgg.h5'))