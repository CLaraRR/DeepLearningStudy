'''
1.cnn + maxpool
2.cnn + maxpool
3.cnn + avgpool
4.softmax + crossentropy
5.在卷积后添加Batch Normalization
'''
from cifar10 import cifar10_input
import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers.python.layers import batch_norm

# get data
batch_size = 128
data_dir = "./cifar10"
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

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def avg_pool_6x6(x):
    return tf.nn.avg_pool(x, ksize=[1, 6, 6, 1], strides=[1, 6, 6, 1], padding="SAME")


def batch_norm_layer(value, is_train, name = 'batch_norm'):
    if is_train is True:
        return batch_norm(value, decay = 0.9, updates_collections = None, is_training = True)
    else:
        return batch_norm(value, decay = 0.9, updates_collections = None, is_training = False)


# 定义占位符
X = tf.placeholder(tf.float32, [None, 24, 24, 3])  # cifar10的图片的shape是24*24*3
Y = tf.placeholder(tf.float32, [None, 10])  # 0-9数字分类-->10 classes
is_train = tf.placeholder(tf.bool)

# 定义第一个卷积层、最大池化层
W_conv1 = weight_variable([5, 5, 3, 64])  # 定义卷积核的高度、宽度、图像通道数、输出的feature map个数
b_conv1 = bias_variable([64])  # 定义偏置值的个数和卷积核的输出个数一样

x_image = tf.reshape(X, [-1, 24, 24, 3])

h_conv1 = tf.nn.relu(batch_norm_layer(conv2d(x_image, W_conv1) + b_conv1, is_train = is_train))  # 原始图像经过卷积操作，然后还要用relu函数来激活
h_pool1 = max_pool_2x2(h_conv1)  # 激活后进入最大池化层

# 定义第二个卷积层、最大池化层
W_conv2 = weight_variable([5, 5, 64, 64])  # 定义卷积核的高度、宽度、图像通道数（上一层输出个数）、输出的feature map个数
b_conv2 = bias_variable([64])  # 定义偏置值的个数和卷积核的输出个数一样

h_conv2 = tf.nn.relu(batch_norm_layer(conv2d(h_pool1, W_conv2) + b_conv2, is_train = is_train))  # 上一层的输出经过卷积操作，然后还要用relu函数来激活
h_pool2 = max_pool_2x2(h_conv2)  # 激活后进入最大池化层

# 定义第三个（最后）卷积层、均值池化层
W_conv3 = weight_variable([5, 5, 64, 10])  # 定义卷积核的高度、宽度、图像通道数（上一层输出个数）、输出的feature map个数（类别数）
b_conv3 = bias_variable([10])  # 定义偏置值的个数和卷积核的输出个数一样

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)  # 上一层的输出经过卷积操作，然后还要用relu函数来激活
nt_hpool3 = avg_pool_6x6(h_conv3)  # 激活后进入均值池化层
nt_hpool3_flat = tf.reshape(nt_hpool3, [-1, 10])
y_conv = tf.nn.softmax(nt_hpool3_flat)  # 输出节点用softmax激活，得到的y_conv即为预测值

# 定义反向传播结构
# 使用softmax + cross entropy
cost = tf.nn.softmax_cross_entropy_with_logits(logits=nt_hpool3_flat, labels=Y)
# cost = -tf.reduce_sum(y * tf.log(y_conv))

global_step = tf.Variable(0, trainable = False)
decaylearning_rate = tf.train.exponential_decay(0.04, global_step, 1000, 0.9)
optimizer = tf.train.AdamOptimizer(decaylearning_rate).minimize(cost, global_step = global_step)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(Y,
                                                              1))  # tf.argmax返回onehot编码中数值为1的那个元素的下标，也就是类别，tf.equal函数判断两个数是否相等,若相等则返回1，否则返回0
# 计算准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 训练模型
training_epochs = 15000
display_step = 200
saver = tf.train.Saver(max_to_keep=1)
savedir = "./model6/"
sess = tf.Session()
sess.run(tf.global_variables_initializer())
tf.train.start_queue_runners(sess=sess)
for epoch in range(training_epochs):
    image_batch, label_batch = sess.run([images_train, labels_train])
    label_b = np.eye(10, dtype=float)[label_batch]  # one-hot编码
    sess.run([optimizer], feed_dict={X: image_batch, Y: label_b, is_train: True})
    if epoch % display_step == 0:
        training_accuracy = accuracy.eval(feed_dict={X: image_batch, Y: label_b, is_train: False}, session=sess)
        print("step %d, accuracy:%g" % (epoch, training_accuracy))
    saver.save(sess, savedir + "cnn_cifar10.ckpt", global_step=epoch + 1)

print("finished training!!!")

# 测试模型
image_batch, label_batch = sess.run([images_test, labels_test])  # 从测试集中取数据
label_b = np.eye(10, dtype=float)[label_batch]  # one-hot
print("test accuracy: %g" % accuracy.eval(feed_dict={X: image_batch, Y: label_b, is_train: False}, session=sess))



