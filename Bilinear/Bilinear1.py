# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 16:15:22 2018

@author: 16703
"""
import keras
from keras.initializers import glorot_normal
import os
import cv2
from sklearn.utils import shuffle
from keras.utils import np_utils
import numpy as np
import config
import util
import json
import random
def outer_product(x):
    """
    calculate outer-products of 2 tensors

        args 
            x
                list of 2 tensors
                , assuming each of which has shape = (size_minibatch, total_pixels, size_filter)
    """
    return keras.backend.batch_dot(
                x[0]
                , x[1]
                , axes=[1,1]
            ) / x[0].get_shape().as_list()[1] 

def signed_sqrt(x):
    """
    calculate element-wise signed square root

        args
            x
                a tensor
    """
    return keras.backend.sign(x) * keras.backend.sqrt(keras.backend.abs(x) + 1e-9)

def L2_norm(x, axis=-1):
    """
    calculate L2-norm

        args 
            x
                a tensor
    """
    return keras.backend.l2_normalize(x, axis=axis)


def build_model(
    size_heigth=299
    ,size_width=299
    ,no_class=61
    ,no_last_layer_backbone=-9
    
    ,name_optimizer="sgd"
    ,rate_learning=1.0
    ,rate_decay_learning=0.0
    ,rate_decay_weight=0.0
    
    ,name_initializer="glorot_normal"
    ,name_activation_logits="softmax"
    ,name_loss="categorical_crossentropy"

    ,flg_debug=False
    ,**kwargs
):
    
    keras.backend.clear_session()
    
    print("-------------------------------")
    print("parameters:")
    for key, val in locals().items():
        if not val == None and not key == "kwargs":
            print("\t", key, "=",  val)
    print("-------------------------------")
    
    ### 
    ### load pre-trained model
    ###
    tensor_input = keras.layers.Input(shape=[size_heigth,size_width,3])
    model_detector = keras.applications.vgg16.VGG16(
                            input_tensor=tensor_input
                            , include_top=False
                          ,weights='imagenet'
                        )
    
    print(model_detector.summary())
    ### 
    ### bi-linear pooling
    ###

    # extract features from detector
    x_detector = model_detector.layers[no_last_layer_backbone].output
    shape_detector = model_detector.layers[no_last_layer_backbone].output_shape
    if flg_debug:
        print("shape_detector : {}".format(shape_detector))

    # extract features from extractor , same with detector for symmetry DxD model
    shape_extractor = shape_detector
    x_extractor = x_detector
    if flg_debug:
        print("shape_extractor : {}".format(shape_extractor))
        
    
    # rehape to (minibatch_size, total_pixels, filter_size)
    x_detector = keras.layers.Reshape(
            [
                shape_detector[1] * shape_detector[2] , shape_detector[-1]
            ]
        )(x_detector)
    if flg_debug:
        print("x_detector shape after rehsape ops : {}".format(x_detector.shape))
        
    x_extractor = keras.layers.Reshape(
            [
                shape_extractor[1] * shape_extractor[2] , shape_extractor[-1]
            ]
        )(x_extractor)
    if flg_debug:
        print("x_extractor shape after rehsape ops : {}".format(x_extractor.shape))
        
        
    # outer products of features, output shape=(minibatch_size, filter_size_detector*filter_size_extractor)
    x = keras.layers.Lambda(outer_product)(
        [x_detector, x_extractor]
    )
    if flg_debug:
        print("x shape after outer products ops : {}".format(x.shape))
        
        
    # rehape to (minibatch_size, filter_size_detector*filter_size_extractor)
    x = keras.layers.Reshape([shape_detector[-1]*shape_extractor[-1]])(x)
    if flg_debug:
        print("x shape after rehsape ops : {}".format(x.shape))
        
        
    # signed square-root 
    x = keras.layers.Lambda(signed_sqrt)(x)
    if flg_debug:
        print("x shape after signed-square-root ops : {}".format(x.shape))
        
    # L2 normalization
    x = keras.layers.Lambda(L2_norm)(x)
    if flg_debug:
        print("x shape after L2-Normalization ops : {}".format(x.shape))



    ### 
    ### attach FC-Layer
    ###

    if name_initializer != None:
            name_initializer = eval(name_initializer+"()")
            
    x = keras.layers.Dense(
            units=no_class
            ,kernel_regularizer=keras.regularizers.l2(rate_decay_weight)
            ,kernel_initializer=name_initializer
        )(x)
    if flg_debug:
        print("x shape after Dense ops : {}".format(x.shape))
    tensor_prediction = keras.layers.Activation(name_activation_logits)(x)
    if flg_debug:
        print("prediction shape : {}".format(tensor_prediction.shape))

        

    ### 
    ### compile model
    ###
    model_bilinear = keras.models.Model(
                        inputs=[tensor_input]
                        , outputs=[tensor_prediction]
                    )
    
    
    # fix pre-trained weights
    for layer in model_detector.layers:
        layer.trainable = True
        
        
    # define optimizers
    opt_adam = keras.optimizers.adam(
                    lr=rate_learning
                    , decay=rate_decay_learning
                )
    opt_rms = keras.optimizers.RMSprop(
                    lr=rate_learning
                    , decay=rate_decay_learning
                )
    opt_sgd = keras.optimizers.SGD(
                    lr=rate_learning
                    , decay=rate_decay_learning
                    , momentum=0.9
                    , nesterov=False
                )
    optimizers ={
        "adam":opt_adam
        ,"rmsprop":opt_rms
        ,"sgd":opt_sgd
    }
    
    model_bilinear.compile(
        loss=name_loss
        , optimizer=optimizers[name_optimizer]
        , metrics=["categorical_accuracy"]
    )
    
    
    
    if flg_debug:
        model_bilinear.summary()
    
    return model_bilinear
model = build_model(
            # number of output classes, 200 for CUB200
            no_class = 61

            # pretrained model specification, using VGG16
            # "block5_conv3 "
            ,no_last_layer_backbone = 17
    
            # training parametes
            ,rate_learning=1.0
            ,rate_decay_weight=1e-8
    
            ,flg_debug=True
        )




def rotate(image, angle=15, scale=0.9):
    w = image.shape[1]
    h = image.shape[0]
    #rotate matrix
    M = cv2.getRotationMatrix2D((w/2,h/2), angle, scale)
    #rotate
    image = cv2.warpAffine(image,M,(w,h))
    return image

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
    crop = cv2.resize(flipimg, (299,299))
    return crop

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
            
batch_size=20
trian_img_paths,train_labels = process_test(config.TRAIN_ANNOTATION_FILE,config.TRAIN_DIR)

trian_img_paths1,train_labels1 = process_test(config.VAL_ANNOTATION_FILE,config.VAL_DIR)


def train_model(
        model=None
        ,name_model="BCNN_keras"
        ,gen_dir_train=None
        ,gen_dir_valid=None
        ,max_epoch=50
    ):
    
    path_model = "./model/{}/".format(name_model)
    if not os.path.exists(path_model):
        os.mkdir(path_model)

        

    callback_stopper = keras.callbacks.EarlyStopping(
                            monitor='val_loss'
                            , min_delta=1e-3
                            , patience=10
                            , verbose=0
                            , mode='auto'
                        )
    list_callback = [

        callback_stopper
    ]
            
    hist = model.fit_generator(
                gen_dir_train
                ,samples_per_epoch=32739//20
                ,nb_epoch=max_epoch
                , validation_data=gen_dir_valid
                ,nb_val_samples=4982//20
                ,callbacks=list_callback
                ,workers=3
                ,verbose=1
            )
        
    model.save_weights(
        path_model
            + "E[{}]".format(len(hist.history['val_loss']))
            + "_LOS[{:.3f}]".format(hist.history['val_loss'][-1])
            + ".h5" 
    )
    
    return hist


# change LR
opt_sgd = keras.optimizers.SGD(
                lr=1e-3
                , decay=1e-9
                , momentum=0.9
                , nesterov=False
            )
model.compile(
    loss="categorical_crossentropy"
    , optimizer='Adam'
    , metrics=["accuracy"]
)
model.load_weights('E[1]_LOS[3.286]_ACC[0.187].h5')

hist =train_model(
            model=model
            ,gen_dir_train=data_generator(trian_img_paths,train_labels,batch_size)
            ,gen_dir_valid=data_generator(trian_img_paths1,train_labels1,batch_size)
            ,max_epoch=5

        )



model.save(os.path.join('./', 'my_model_vgg16.h5'))