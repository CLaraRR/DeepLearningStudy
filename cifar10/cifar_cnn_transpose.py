'''
1.cnn + maxpool
2.cnn + maxpool
3.cnn + avgpool
4.softmax + crossentropy
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

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


# def max_pool_2x2(x):
#     return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def max_pool_with_argmax(net, stride):
    _, mask = tf.nn.max_pool_with_argmax(net, ksize = [1, stride, stride, 1], strides = [1, stride, stride, 1], padding = 'SAME')
    mask = tf.stop_gradient(mask)
    net = tf.nn.max_pool(net, ksize = [1, stride, stride, 1], strides = [1, stride, stride, 1], padding = 'SAME')
    return net, mask  # 返回池化结果和每个最大值的位置


def avg_pool_6x6(x):
    return tf.nn.avg_pool(x, ksize=[1, 6, 6, 1], strides=[1, 6, 6, 1], padding="SAME")


def unpool(net, mask, stride):
    ksize = [1, stride, stride, 1]
    input_shape = net.get_shape().as_list()

    output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])

    one_like_mask = tf.ones_like(mask)
    batch_range = tf.reshape(tf.range(output_shape[0], dtype=tf.int64), shape=[input_shape[0], 1, 1, 1])
    b = one_like_mask * batch_range
    y = mask // (output_shape[2] * output_shape[3])
    x = mask % (output_shape[2] * output_shape[3]) // output_shape[3]
    feature_range = tf.range(output_shape[3], dtype=tf.int64)
    f = one_like_mask * feature_range

    updates_size = tf.size(net)
    indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, updates_size]))
    values = tf.reshape(net, [updates_size])
    ret = tf.scatter_nd(indices, values, output_shape)
    return ret


# 定义占位符
X = tf.placeholder(tf.float32, [None, 24, 24, 3])  # cifar10的图片的shape是24*24*3
Y = tf.placeholder(tf.float32, [None, 10])  # 0-9数字分类-->10 classes

# 定义第一个卷积层、最大池化层
W_conv1 = weight_variable([5, 5, 3, 64])  # 定义卷积核的高度、宽度、图像通道数、输出的feature map个数
b_conv1 = bias_variable([64])  # 定义偏置值的个数和卷积核的输出个数一样

x_image = tf.reshape(X, [-1, 24, 24, 3])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # 原始图像经过卷积操作，然后还要用relu函数来激活
h_pool1, mask1 = max_pool_with_argmax(h_conv1, 2)  # 激活后进入最大池化层

# 定义第二个卷积层、最大池化层
W_conv2 = weight_variable([5, 5, 64, 64])  # 定义卷积核的高度、宽度、图像通道数（上一层输出个数）、输出的feature map个数
b_conv2 = bias_variable([64])  # 定义偏置值的个数和卷积核的输出个数一样

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # 上一层的输出经过卷积操作，然后还要用relu函数来激活
h_pool2, mask = max_pool_with_argmax(h_conv2, 2)  # 激活后进入最大池化层
print(h_pool2.shape)

# 从第二层开始反卷积还原图像
t_conv2 = unpool(h_pool2, mask, 2)
t_pool1 = tf.nn.conv2d_transpose(t_conv2 - b_conv2, W_conv2, h_pool1.shape, [1, 1, 1, 1])
print(t_conv2.shape, h_pool1.shape, t_pool1.shape)
t_conv1 = unpool(t_pool1, mask1, 2)
t_x_image = tf.nn.conv2d_transpose(t_conv1 - b_conv1, W_conv1, x_image.shape, [1, 1, 1, 1])

# 从第一层开始反卷积还原图像
t1_conv1 = unpool(h_pool1, mask1, 2)
t1_x_image = tf.nn.conv2d_transpose(t1_conv1 - b_conv1, W_conv1, x_image.shape, [1, 1, 1, 1])

# 合并还原结果，输出给tensorboard进行显示
# 生成最终图像
stitched_decodings = tf.concat((x_image, t1_x_image, t_x_image), axis = 2)
decoding_summary_op = tf.summary.image("source/cifar", stitched_decodings)



# 定义第三个（最后）卷积层、均值池化层
W_conv3 = weight_variable([5, 5, 64, 10])  # 定义卷积核的高度、宽度、图像通道数（上一层输出个数）、输出的feature map个数（类别数）
b_conv3 = bias_variable([10])  # 定义偏置值的个数和卷积核的输出个数一样

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)  # 上一层的输出经过卷积操作，然后还要用relu函数来激活
nt_hpool3 = avg_pool_6x6(h_conv3)  # 激活后进入均值池化层
nt_hpool3_flat = tf.reshape(nt_hpool3, [-1, 10])
y_conv = tf.nn.softmax(nt_hpool3_flat)  # 输出节点用softmax激活，得到的y_conv即为预测值

# 定义反向传播结构
# 使用softmax + cross entropy + L2正则化
cost = -tf.reduce_sum(Y * tf.log(y_conv)) + (tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(W_conv2) + tf.nn.l2_loss(W_conv3))


learning_rate = 1e-4
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(Y, 1))  # tf.argmax返回onehot编码中数值为1的那个元素的下标，也就是类别，tf.equal函数判断两个数是否相等,若相等则返回1，否则返回0
# 计算准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 训练模型
training_epochs = 15000
display_step = 200
saver = tf.train.Saver(max_to_keep=1)
savedir = "./model3/"
sess = tf.Session()
sess.run(tf.global_variables_initializer())
summary_writer = tf.summary.FileWriter("./log_cnn_transpose/", sess.graph)
tf.train.start_queue_runners(sess=sess)
for epoch in range(training_epochs):
    image_batch, label_batch = sess.run([images_train, labels_train])
    label_b = np.eye(10, dtype=float)[label_batch]  # one-hot编码
    sess.run([optimizer], feed_dict={X: image_batch, Y: label_b})
    if epoch % display_step == 0:
        training_accuracy = accuracy.eval(feed_dict={X: image_batch, Y: label_b}, session=sess)
        print("step %d, accuracy:%g" % (epoch, training_accuracy))
    saver.save(sess, savedir + "cnn_cifar10.ckpt", global_step=epoch + 1)

print("finished training!!!")

# 测试模型
image_batch, label_batch = sess.run([images_test, labels_test])  # 从测试集中取数据
label_b = np.eye(10, dtype=float)[label_batch]  # one-hot
print("test accuracy: %g" % accuracy.eval(feed_dict={X: image_batch, Y: label_b}, session=sess))

decoding_summary = sess.run(decoding_summary_op, feed_dict = {X: image_batch, Y: label_b})
summary_writer.add_summary(decoding_summary)

