# -*- coding: utf-8 -*-
import tensorflow as tf
import sys
import config
import util
import json
import numpy as np
from sklearn.utils import shuffle
import cv2
size=128
batch_size=16
def load_feature(img_path):
    img = util.cv_imread(img_path)
    norm_img= img/255.
    resized_img = util.resize_img(norm_img,config.INPUT_SIZE)
    crop = cv2.resize(resized_img, (config.INPUT_SIZE, config.INPUT_SIZE))
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
    img_paths,labels = shuffle(img_paths,labels)
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
            


trian_img_paths,train_labels,qname = process_test(config.TRAIN_ANNOTATION_FILE,config.TRAIN_DIR)
trian_img_paths1,train_labels1,qname1 = process_test(config.VAL_ANNOTATION_FILE,config.VAL_DIR)
train_data_gen = data_generator(trian_img_paths,train_labels,qname,batch_size)
train_data_gen1 =data_generator(trian_img_paths1,train_labels1,qname1,batch_size)
num_batch = len(train_labels) // batch_size

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

def cnnTrain():
    out = cnnLayer()
    saver = tf.train.Saver()  
   
    
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y_))

    train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
    # 比较标签是否相等，再求的所有数的平均值，tf.cast(强制转换类型)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(y_, 1)), tf.float32))
    # 将loss与accuracy保存以供tensorboard使用

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        saver.restore(sess, tf.train.latest_checkpoint('./model')) 
        n=0
        
        while 1:
            batch_features, batch_labels,batch_name = next(train_data_gen)

                # 开始训练数据，同时训练三个变量，返回三个数据
            _,loss = sess.run([train_step, cross_entropy],
                                           feed_dict={x:batch_features,y_:batch_labels, keep_prob_5:0.5,keep_prob_75:0.75})

                # 打印损失
          
            print(n,loss)
            n+=1
            if (n) % 100 == 0:
                batch_features1, batch_labels1,batch_name1 = next(train_data_gen1)
                    # 获取测试数据的准确率
                acc = accuracy.eval({x:batch_features1, y_:batch_labels1, keep_prob_5:1.0, keep_prob_75:1.0})
                print(n, acc,)
                
                    # 准确率大于0.98时保存并退出
                if acc > 0.90 or n > 5000:
                    saver.save(sess, './model/finetun/train_faces.model', global_step=n)
                        #builder = tf.saved_model.builder.SavedModelBuilder('./model/'+'savemodel')

                        #builder.add_meta_graph_and_variables(sess,['cpu_server_1'])
                        #builder.save()  # 保存 PB 
                    print ('saver done')
                    sys.exit(0)
        

if __name__ == '__main__':
    cnnTrain()
