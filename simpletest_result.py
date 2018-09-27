# -*- coding: utf-8 -*-
import tensorflow as tf
from collections import OrderedDict
import util
import json
import numpy as np
import os
import cv2
size=128
batch_size=1
def load_feature(img_path):
    img = util.cv_imread(img_path)
    norm_img= img/255.
    resized_img = util.resize_img(norm_img,size)
    crop = cv2.resize(resized_img, (size,size))
    return crop
def process_test(anno_file,dir):
    with open(anno_file) as file:
        annotations = json.load(file)
        img_paths = []
        labels = []
        name=[]
        for anno in annotations:
            img_paths.append(dir + anno["image_id"])
            name.append(anno["image_id"])
            labels.append(anno["disease_class"])
    return img_paths, labels,name

def data_generator(img_paths,labels,name,batch_size):
    num_sample = len(img_paths)
    print(num_sample)
    while True:
        for offset in range(0,num_sample,batch_size):
            batch_paths = img_paths[offset:offset+batch_size]
            batch_labels = labels[offset:offset+batch_size]
            batch_names= name[offset:offset+batch_size]
            batch_labels=np.array(batch_labels)
            batch_features = [load_feature(path) for path in batch_paths]
            batch_labels =util.make_one_hot(batch_labels)
            batch_feature = np.array(batch_features)
            yield batch_feature, batch_labels,np.array(batch_names)
            
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
            batch_features = [util.load_feature(path) for path in batch_paths]
            batch_feature = np.array(batch_features)
            yield batch_feature, np.array(batch_names)


#trian_img_paths,train_labels,name = process_test(config.VAL_ANNOTATION_FILE,config.VAL_DIR)
#train_data_gen = data_generator(trian_img_paths,train_labels,name,batch_size)
#num_batch = len(train_labels) // batch_size

x = tf.placeholder(tf.float32, [None, size, size, 3])
y_ = tf.placeholder(tf.float32, [None, 61])

keep_prob_5 = tf.placeholder(tf.float32)
keep_prob_75 = tf.placeholder(tf.float32)

#权重
def weightVariable(shape):
    init = tf.random_normal(shape, stddev=0.01)
    return tf.Variable(init)
#偏置
def biasVariable(shape):
    init = tf.random_normal(shape)
    return tf.Variable(init)
#卷积层
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
#
def maxPool(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def dropout(x, keep):
    return tf.nn.dropout(x, keep)

def cnnLayer():
    # 第一层
    W1 = weightVariable([3,3,3,32]) # 卷积核大小(3,3)， 输入通道(3)， 输出通道(32)
    b1 = biasVariable([32])
    # 卷积
    conv1 = tf.nn.relu(conv2d(x, W1) + b1)
    # 池化
    pool1 = maxPool(conv1)
    # 减少过拟合，随机让某些权重不更新
    drop1 = dropout(pool1, keep_prob_5)

    # 第二层
    W2 = weightVariable([3,3,32,64])
    b2 = biasVariable([64])
    conv2 = tf.nn.relu(conv2d(drop1, W2) + b2)
    pool2 = maxPool(conv2)
  

    # 第三层
    W3 = weightVariable([3,3,64,64])
    b3 = biasVariable([64])
    conv3 = tf.nn.relu(conv2d(pool2, W3) + b3)
    pool3 = maxPool(conv3)
    

    W4 = weightVariable([3,3,64,128])
    b4 = biasVariable([128])
    conv4 = tf.nn.relu(conv2d(pool3, W4) + b4)
    pool4 = maxPool(conv4)
    drop4 = dropout(pool4, keep_prob_5)
    
    # 全连接层
    Wf = weightVariable([8*8*128, 512])
    bf = biasVariable([512])
    drop3_flat = tf.reshape(drop4, [-1, 8*8*128])
    dense = tf.nn.relu(tf.matmul(drop3_flat, Wf) + bf)
    dropf = dropout(dense, keep_prob_75)

    # 输出层
    Wout = weightVariable([512,61])
    bout = weightVariable([61])
    #out = tf.matmul(dropf, Wout) + bout
    out = tf.add(tf.matmul(dropf, Wout), bout)
    return out



output = cnnLayer()  
predict = tf.argmax(output, 1)  
   
saver = tf.train.Saver()  
sess = tf.Session()  
saver.restore(sess, tf.train.latest_checkpoint('./model'))  

nums=[]
file_name = './json_file.json'
#test_dir = "./AgriculturalDisease_validationset/images"
test_dir = "./AgriculturalDisease_testA/images"
imgs,names =datasettest(test_dir)
num_batch = len(names) // batch_size
train_data_gen = datatest_generator(imgs,names,batch_size)

while True:
    batch_features,tnames = next(train_data_gen)
    tnames = ''.join(tnames)
    #batch_features, batch_labels,names = next(train_data_gen)
    #res = sess.run(predict,feed_dict={x:batch_features,y_:batch_labels, keep_prob_5:1.0,keep_prob_75:1.0})
    res = sess.run(predict,feed_dict={x:batch_features, keep_prob_5:1.0,keep_prob_75:1.0})
    res=res[0]
    
    num={"image_id":str(tnames),"disease_class":int(res)}
    OrderedDict(sorted(num.items(),key=lambda t: t[0]),reverse=True)
    nums.append(num)
    

    if tnames == 'u=3337154909,2169261495&fm=27&gp=0.jpg': 
    #if tnames ==   'u=2520622969,1450176384&fm=200&gp=0.jpg':
        jsonData = json.dumps(nums)
        fileObject = open('json_file.json', 'w')
        fileObject.write(jsonData)
        fileObject.close()
    
        break
with open(file_name, 'r') as file_obj:
            '''写入json文件'''
            n=json.load(file_obj)
            print(n)
            