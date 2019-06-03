'''
1.labels format: not one-hot encoding
2.network: maxout network
3.loss: softmax cross entropy
4.optimizer: gradient descent
'''
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
import pylab
import warnings
warnings.filterwarnings('ignore')


# 使用非ont-hot编码的标签
mnist = input_data.read_data_sets("./MNIST_data/")

# im = mnist.train.images[1]
# im = im.reshape(-1, 28)
# pylab.imshow(im)
# pylab.show()
# print(mnist.test.images.shape)
# print(mnist.validation.images.shape)


def max_out(inputs, num_units, axis=None):
    shape = inputs.get_shape().as_list()
    if shape[0] is None:
        shape[0] = -1
    if axis is None:  # Assume that channel is the last dimension
        axis = -1
    num_channels = shape[axis]
    if num_channels % num_units:
        raise ValueError('number of features({}) is not '
                         'a multiple of num_units({})'.format(num_channels, num_units))
    shape[axis] = num_units
    shape += [num_channels // num_units]
    outputs = tf.reduce_max(tf.reshape(inputs, shape), -1, keep_dims=False)
    return outputs


tf.reset_default_graph()

# 定义正向传播的结构
# 定义占位符
x = tf.placeholder(tf.float32, [None, 784])  # MNIST数据集的维度是28*28=784
y = tf.placeholder(tf.int32, [None])  # 数字0~9，共10个类别，因为标签已经不是one-hot编码所以占位符只是一个数字，代表哪一类,要注意这里的数据类型必须是int32或int64
# 定义学习参数
W = tf.Variable(tf.random_normal([784, 100]), tf.float32)
b = tf.Variable(tf.zeros([100]), tf.float32)

# 定义输出节点
z = tf.matmul(x, W) + b
maxout = max_out(z, 50) 

W2 = tf.Variable(tf.truncated_normal([50, 10], stddev = 0.1))
b2 = tf.Variable(tf.zeros([10]))

z2 = tf.matmul(maxout, W2) + b2
pred = tf.nn.softmax(z2)



# 定义反向传播的结构
# 损失函数,输入数据直接经过tf.nn.sparse_softmax_cross_entropy_with_logits函数得到loss
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = z2, labels = y ))
# 定义参数
learning_rate = 0.04
# 使用梯度下降优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


# 训练模型
# 初始化所有变量,所有变量的定义一定都要放在这条语句之前
init = tf.global_variables_initializer()
# 定义参数
training_epochs = 200  # 迭代次数
batch_size = 100  # 训练过程中一次取多少条数据进行训练，一批次多少条数据
display_step = 2  # 没训练一次就把具体的中间状态显示出来
saver = tf.train.Saver(max_to_keep= 1)
savedir = "./model_maxout/"
with tf.Session() as sess:
    sess.run(init)
    # 启动循环开始训练
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples/batch_size)  # 计算得到一共要进行多少批次
        # 循环所有数据集
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)  # 获得下一个批次的训练数据,数据是随机的
            # 运行优化器
            _, c = sess.run([optimizer, cost], feed_dict = {x: batch_xs, y: batch_ys})
            # 计算平均loss值,??????
            avg_cost += c/total_batch
        # 显示每一次训练完后的详细信息
        if (epoch + 1) % display_step == 0:
            print("Epoch:", epoch + 1, "cost=", avg_cost)
        saver.save(sess, savedir + "maxout_mnist.ckpt", global_step = epoch + 1)
    print("Finished!!!")
    # 测试模型
    correct_prediction = tf.equal(tf.cast(tf.argmax(pred, 1), tf.int32), y)  # tf.argmax返回onehot编码中数值为1的那个元素的下标，也就是类别，tf.equal函数判断两个数是否相等,若相等则返回1，否则返回0
    # 计算准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # print("Accuracy:", sess.run(accuracy, feed_dict = {x: mnist.test.images, y: mnist.test.labels}))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


    
    
    




