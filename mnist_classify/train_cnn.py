'''
(one-hot encoding)
1.cnn + maxpool
2.cnn + maxpool
3.cnn + avgpool
4.softmax + crossentropy
'''
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
import pylab


# 使用one-hot编码的标签
mnist = input_data.read_data_sets("./mnist_classify/MNIST_data/", one_hot = True)

# im = mnist.train.images[1]
# im = im.reshape(-1, 28)
# pylab.imshow(im)
# pylab.show()
# print(mnist.test.images.shape)
# print(mnist.validation.images.shape)


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
  
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')  
def avg_pool_7x7(x):
  return tf.nn.avg_pool(x, ksize=[1, 7, 7, 1],
                        strides=[1, 7, 7, 1], padding='SAME')


tf.reset_default_graph()

# 定义正向传播的结构
# 定义占位符
X = tf.placeholder(tf.float32, [None, 784])  # MNIST数据集的维度是28*28=784
Y = tf.placeholder(tf.float32, [None, 10])  # 数字0~9，共10个类别
# 定义第一个卷积层、最大池化层
W_conv1 = weight_variable([5, 5, 1, 32])  # 定义卷积核的高度、宽度、图像通道数、输出的feature map个数
b_conv1 = bias_variable([32]) # 定义偏置值的个数和卷积核的输出个数一样

x_image = tf.reshape(X, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # 原始图像经过卷积操作，然后还要用relu函数来激活
h_pool1 = max_pool_2x2(h_conv1)  # 激活后进入最大池化层
# 定义第二个卷积层、最大池化层
W_conv2 = weight_variable([5, 5, 32, 64])  # 定义卷积核的高度、宽度、图像通道数（上一层输出个数）、输出的feature map个数
b_conv2 = bias_variable([64]) # 定义偏置值的个数和卷积核的输出个数一样

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # 上一层的输出经过卷积操作，然后还要用relu函数来激活
h_pool2 = max_pool_2x2(h_conv2)  # 激活后进入最大池化层
# 定义第三个（最后）卷积层、均值池化层
W_conv3 = weight_variable([5, 5, 64, 10])  # 定义卷积核的高度、宽度、图像通道数（上一层输出个数）、输出的feature map个数（类别数）
b_conv3 = bias_variable([10]) # 定义偏置值的个数和卷积核的输出个数一样
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)  # 上一层的输出经过卷积操作，然后还要用relu函数来激活

nt_hpool3=avg_pool_7x7(h_conv3)  # 激活后进入均值池化层
nt_hpool3_flat = tf.reshape(nt_hpool3, [-1, 10])
y_conv=tf.nn.softmax(nt_hpool3_flat)  # 输出节点用softmax激活，得到的y_conv即为预测值



# 定义反向传播的结构
# 使用softmax + cross entropy
cost = tf.nn.softmax_cross_entropy_with_logits(logits = nt_hpool3_flat, labels = Y)
# cost = -tf.reduce_sum(y * tf.log(y_conv))

learning_rate = 1e-4
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(Y, 1))  # tf.argmax返回onehot编码中数值为1的那个元素的下标，也就是类别，tf.equal函数判断两个数是否相等,若相等则返回1，否则返回0
# 计算准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# 训练模型
# 初始化所有变量,所有变量的定义一定都要放在这条语句之前
init = tf.global_variables_initializer()
# 定义参数
training_epochs = 20000  # 迭代次数
batch_size = 50  # 训练过程中一次取多少条数据进行训练，一批次多少条数据
display_step = 2  # 没训练一次就把具体的中间状态显示出来
saver = tf.train.Saver(max_to_keep= 1)
savedir = "./mnist_classify/model_cnn/"
with tf.Session() as sess:
    sess.run(init)
    # 启动循环开始训练
    for epoch in range(training_epochs):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)  # 获得下一个批次的训练数据,数据是随机的
        # 运行优化器
        sess.run([optimizer], feed_dict = {X: batch_xs, Y: batch_ys})
        # 显示每一次训练完后的详细信息
        if (epoch + 1) % display_step == 0:
            training_accuracy = accuracy.eval(feed_dict = {X: batch_xs, Y: batch_ys}, session = sess)
            print("step %d, accuracy:%g" % (epoch, training_accuracy))
        saver.save(sess, savedir + "cnn_mnist.ckpt", global_step = epoch + 1)
    print("Finished!!!")
    # 测试模型
    print ("test accuracy: %g" % accuracy.eval(feed_dict = {X: mnist.test.images, Y: mnist.test.labels}))


    
    
    




