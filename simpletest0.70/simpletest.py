import tensorflow as tf
import sys
import config
import util


size=128
batch_size=16


trian_img_paths,train_labels = util.process_annotation(config.TRAIN_ANNOTATION_FILE,config.TRAIN_DIR)
train_data_gen = util.data_generator(trian_img_paths,train_labels,batch_size)
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
   
    
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y_))

    train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
    # 比较标签是否相等，再求的所有数的平均值，tf.cast(强制转换类型)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(y_, 1)), tf.float32))
    # 将loss与accuracy保存以供tensorboard使用
 
    
    saver = tf.train.Saver()

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        n=0
        
        while 1:
            batch_features, batch_labels = next(train_data_gen)

                # 开始训练数据，同时训练三个变量，返回三个数据
            _,loss = sess.run([train_step, cross_entropy],
                                           feed_dict={x:batch_features,y_:batch_labels, keep_prob_5:0.5,keep_prob_75:0.75})

                # 打印损失
            print(n, loss)
            n+=1
            if (n) % 100 == 0:
                    # 获取测试数据的准确率
                acc = accuracy.eval({x:batch_features, y_:batch_labels, keep_prob_5:1.0, keep_prob_75:1.0})
                print(n, acc,)
                    # 准确率大于0.98时保存并退出
                if acc > 0.90 or n >10000:
                    saver.save(sess, './model/train_faces.model', global_step=n)
                        #builder = tf.saved_model.builder.SavedModelBuilder('./model/'+'savemodel')

                        #builder.add_meta_graph_and_variables(sess,['cpu_server_1'])
                        #builder.save()  # 保存 PB 
                    print ('saver done')
                    sys.exit(0)
        

if __name__ == '__main__':
    cnnTrain()