'''
1.cnn + maxpool
2.cnn + maxpool
3.fullyconnected network
4.softmax + crossentropy
'''
from cifar10 import cifar10_input
import tensorflow as tf 
import numpy as np
import pylab 

# get data
batch_size = 128
data_dir = "./cifar_10"
images_train, labels_train = cifar10_input.inputs(eval_data = False,  batch_size = batch_size)
images_test , labels_test =  cifar10_input.inputs(eval_data = True, batch_size = batch_size)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
tf.train.start_queue_runners()
image_batch, label_batch = sess.run([images_test, labels_test])
# print(image_batch[0])
# print(label_batch[0])
# pylab.imshow(image_batch[0])
# pylab.show()

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

def avg_pool_6x6(x):
    return tf.nn.avg_pool(x, ksize = [1, 6, 6, 1], strides = [1, 6, 6, 1], padding = "SAME")


# 定义占位符
X = tf.placeholder(tf.float32, [None, 24, 24, 3])  # cifar10的图片的shape是24*24*3
Y = tf.placeholder(tf.float32, [None, 10])  # 0-9数字分类-->10 classes

# 定义第一个卷积层、最大池化层
W_conv1 = weight_variable([5, 5, 3, 64])  # 定义卷积核的高度、宽度、图像通道数、输出的feature map个数
b_conv1 = bias_variable([64]) # 定义偏置值的个数和卷积核的输出个数一样

x_image = tf.reshape(X, [-1, 24, 24, 3])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # 原始图像经过卷积操作，然后还要用relu函数来激活
h_pool1 = max_pool_2x2(h_conv1)  # 激活后进入最大池化层

# 定义第二个卷积层、最大池化层
W_conv2 = weight_variable([5, 5, 64, 64])  # 定义卷积核的高度、宽度、图像通道数（上一层输出个数）、输出的feature map个数
b_conv2 = bias_variable([64]) # 定义偏置值的个数和卷积核的输出个数一样

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # 上一层的输出经过卷积操作，然后还要用relu函数来激活
h_pool2 = max_pool_2x2(h_conv2)  # 激活后进入最大池化层

# 定义全连接层
W_fc1 = weight_variable([6 * 6 * 64, 256])
b_fc1 = bias_variable([256])

h_pool2_flat = tf.reshape(h_pool2, [-1, 6*6*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, rate = 1 - keep_prob)

W_fc2 = weight_variable([256, 128])
b_fc2 = bias_variable([128])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

W_fc3 = weight_variable([128, 10])
b_fc3 = bias_variable([10])
z = tf.matmul(h_fc2, W_fc3) + b_fc3
y_conv=tf.nn.softmax(z)


# 定义反向传播结构
# 使用softmax + cross entropy
cost = tf.nn.softmax_cross_entropy_with_logits(logits = z, labels = Y)
# cost = -tf.reduce_sum(y * tf.log(y_conv))

learning_rate = 1e-4
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(Y, 1))  # tf.argmax返回onehot编码中数值为1的那个元素的下标，也就是类别，tf.equal函数判断两个数是否相等,若相等则返回1，否则返回0
# 计算准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



# 训练模型
training_epochs = 15000
display_step = 2
saver = tf.train.Saver(max_to_keep= 1)
savedir = "./model2/"
sess = tf.Session()
sess.run(tf.global_variables_initializer())
tf.train.start_queue_runners(sess = sess)
for epoch in range(training_epochs):
    image_batch, label_batch = sess.run([images_train, labels_train])
    label_b = np.eye(10, dtype = float)[label_batch]  # one-hot编码
    sess.run([optimizer], feed_dict = {X: image_batch, Y: label_b, keep_prob: 0.5})
    if epoch % display_step == 0:
        training_accuracy = accuracy.eval(feed_dict = {X: image_batch, Y: label_b, keep_prob: 1.0}, session = sess)
        print("step %d, accuracy:%g" % (epoch, training_accuracy))
    saver.save(sess, savedir + "cnn2_cifar10.ckpt", global_step = epoch + 1)
    

print("finished training!!!")

# 测试模型
image_batch, label_batch = sess.run([images_test, labels_test])  # 从测试集中取数据
label_b = np.eye(10, dtype = float)[label_batch]  # one-hot
print ("test accuracy: %g" % accuracy.eval(feed_dict = {X: image_batch, Y: label_b, keep_prob: 1.0}, session = sess))



