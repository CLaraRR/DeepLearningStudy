'''
1.cnn + maxpool
2.cnn + maxpool
3.fullyconnected network
4.softmax + crossentropy
5.用高级API tf.contrib.layers.conv2d替代tf.nn.conv2d
'''
from cifar10 import cifar10_input
import tensorflow as tf
import numpy as np
import pylab

# get data
batch_size = 128
data_dir = "./cifar_10"
images_train, labels_train = cifar10_input.inputs(eval_data=False, batch_size=batch_size)
images_test, labels_test = cifar10_input.inputs(eval_data=True, batch_size=batch_size)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
tf.train.start_queue_runners()
image_batch, label_batch = sess.run([images_test, labels_test])


# print(image_batch[0])
# print(label_batch[0])
# pylab.imshow(image_batch[0])
# pylab.show()


# 定义占位符
X = tf.placeholder(tf.float32, [None, 24, 24, 3])  # cifar10的图片的shape是24*24*3
Y = tf.placeholder(tf.float32, [None, 10])  # 0-9数字分类-->10 classes

# 定义第一个卷积层、最大池化层
x_image = tf.reshape(X, [-1, 24, 24, 3])

h_conv1 = tf.contrib.layers.conv2d(x_image, 64, 5, 1, 'SAME', activation_fn = tf.nn.relu)  # 原始图像经过卷积操作，然后还要用relu函数来激活
h_pool1 = tf.contrib.layers.max_pool2d(h_conv1, [2, 2], stride = 2, padding = 'SAME')  # 激活后进入最大池化层

# 定义第二个卷积层、最大池化层
h_conv2 = tf.contrib.layers.conv2d(h_pool1, 64, [5, 5], 1, 'SAME', activation_fn = tf.nn.relu)  # 上一层的输出经过卷积操作，然后还要用relu函数来激活
h_pool2 = tf.contrib.layers.max_pool2d(h_conv2, [2, 2], stride = 2, padding = 'SAME')  # 激活后进入最大池化层

nt_hpool2 = tf.contrib.layers.avg_pool2d(h_pool2, [6, 6], stride = 6, padding = 'SAME')
nt_hpool2_flat = tf.reshape(nt_hpool2, [-1, 64])

# 定义全连接层
y_conv = tf.contrib.layers.fully_connected(nt_hpool2_flat, 10, activation_fn = tf.nn.softmax)


# 定义反向传播结构
# 使用softmax + cross entropy
cost = - tf.reduce_sum(Y * tf.log(y_conv))

learning_rate = 1e-4
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(Y, 1))  # tf.argmax返回onehot编码中数值为1的那个元素的下标，也就是类别，tf.equal函数判断两个数是否相等,若相等则返回1，否则返回0
# 计算准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 训练模型
training_epochs = 15000
display_step = 2
saver = tf.train.Saver(max_to_keep=1)
savedir = "./model4/"
sess = tf.Session()
sess.run(tf.global_variables_initializer())
tf.train.start_queue_runners(sess=sess)
for epoch in range(training_epochs):
    image_batch, label_batch = sess.run([images_train, labels_train])
    label_b = np.eye(10, dtype=float)[label_batch]  # one-hot编码
    sess.run([optimizer], feed_dict={X: image_batch, Y: label_b})
    if epoch % display_step == 0:
        training_accuracy = accuracy.eval(feed_dict={X: image_batch, Y: label_b}, session=sess)
        print("step %d, accuracy:%g" % (epoch, training_accuracy))
    saver.save(sess, savedir + "cnn2_cifar10.ckpt", global_step=epoch + 1)

print("finished training!!!")

# 测试模型
image_batch, label_batch = sess.run([images_test, labels_test])  # 从测试集中取数据
label_b = np.eye(10, dtype=float)[label_batch]  # one-hot
print("test accuracy: %g" % accuracy.eval(feed_dict={X: image_batch, Y: label_b}, session=sess))



