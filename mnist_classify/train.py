'''
1.one-hot encoding
2.softmax cross entropy
3.
'''
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
import pylab


# 使用one-hot编码的标签
mnist = input_data.read_data_sets("./MNIST_data/", one_hot = True)

# im = mnist.train.images[1]
# im = im.reshape(-1, 28)
# pylab.imshow(im)
# pylab.show()
# print(mnist.test.images.shape)
# print(mnist.validation.images.shape)

tf.reset_default_graph()

# 定义正向传播的结构
# 定义占位符
x = tf.placeholder(tf.float32, [None, 784])  # MNIST数据集的维度是28*28=784
y = tf.placeholder(tf.float32, [None, 10])  # 数字0~9，共10个类别
# 定义学习参数
W = tf.Variable(tf.random_normal([784, 10]), tf.float32)
b = tf.Variable(tf.zeros([10]), tf.float32)
# 定义输出节点，选择激活函数为softmax，此时可得到一个predict probability向量
pred = tf.nn.softmax(tf.matmul(x, W) + b)  # softmax分类



# 定义反向传播的结构
# 损失函数,输入数据经过上面的softmax函数激活后再经过交叉熵cross entropy，从而得到loss
# cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices = 1))
# 或者直接使用tf.softmax_cross_entropy_with_logits函数来达到softmax + cross entropy的效果
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = tf.matmul(x, W) + b, labels = y))  
# 定义参数
learning_rate = 0.01
# 使用梯度下降优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


# 训练模型
# 初始化所有变量,所有变量的定义一定都要放在这条语句之前
init = tf.global_variables_initializer()
# 定义参数
training_epochs = 25  # 迭代次数
batch_size = 100  # 训练过程中一次取多少条数据进行训练，一批次多少条数据
display_step = 1  # 没训练一次就把具体的中间状态显示出来
saver = tf.train.Saver(max_to_keep= 1)
savedir = "./model/"
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
        # saver.save(sess, savedir + "softmax_mnist.ckpt", global_step = epoch + 1)
    print("Finished!!!")
    # 测试模型
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))  # tf.argmax返回onehot编码中数值为1的那个元素的下标，也就是类别，tf.equal函数判断两个数是否相等,若相等则返回1，否则返回0
    # 计算准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # print("Accuracy:", sess.run(accuracy, feed_dict = {x: mnist.test.images, y: mnist.test.labels}))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


    
    
    




