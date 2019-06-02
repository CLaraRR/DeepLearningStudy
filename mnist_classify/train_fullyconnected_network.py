import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
import pylab

# 使用one-hot编码的标签
mnist = input_data.read_data_sets("./mnist_classify/MNIST_data/", one_hot = True)

tf.reset_default_graph()

# 定义网络结构
# 设置网络模型参数
n_hidden_1 = 256  # 第一个隐藏层节点个数
n_hidden_2 = 256  # 第二个隐藏层节点个数
n_input = 784  # MNIST共784维（28*28=784）
n_classes = 10  # MNIST共10个类别
# 定义占位符
X = tf.placeholder(tf.float32, [None, n_input])
Y = tf.placeholder(tf.float32, [None, n_classes])
# 创建model
def multilayer_perceptron(x, weights, biases):
    # 第一个隐藏层
    z1 = tf.matmul(x, weights["h1"]) + biases["h1"]
    layer_1 = tf.nn.relu(z1)
    # 第二个隐藏层
    z2 = tf.matmul(layer_1, weights["h2"]) + biases["h2"]
    layer_2 = tf.nn.relu(z2)
    # 输出层
    output_layer = tf.matmul(layer_2, weights["out"]) + biases["out"]
    return output_layer  # 返回的outputval不经过激活函数直接返回

# 学习参数
weights = {
    "h1": tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    "h2": tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    "out": tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    "h1": tf.Variable(tf.random_normal([n_hidden_1])),
    "h2": tf.Variable(tf.random_normal([n_hidden_2])),
    "out": tf.Variable(tf.random_normal([n_classes]))
}

# 输出值
pred = multilayer_perceptron(X, weights, biases)
# 定义loss和optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = Y))
learning_rate = 0.001
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


# 训练
training_epochs = 25
batch_size = 100
display_step = 1

saver = tf.train.Saver(max_to_keep= 1)
savedir = "./mnist_classify/model_fc/"
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 启动循环开始训练
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples/batch_size)  # 计算得到一共要进行多少批次
        # 循环所有数据集
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)  # 获得下一个批次的训练数据,数据是随机的
            # 运行优化器
            _, c = sess.run([optimizer, cost], feed_dict = {X: batch_xs, Y: batch_ys})
            # 计算平均loss值,??????
            avg_cost += c/total_batch
        # 显示每一次训练完后的详细信息
        if (epoch + 1) % display_step == 0:
            print("Epoch:", epoch + 1, "cost=", avg_cost)
        saver.save(sess, savedir + "fc_mnist.ckpt", global_step = epoch + 1)
    print("Finished!!!")
    # 测试模型
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))  # tf.argmax返回onehot编码中数值为1的那个元素的下标，也就是类别，tf.equal函数判断两个数是否相等,若相等则返回1，否则返回0
    # 计算准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # print("Accuracy:", sess.run(accuracy, feed_dict = {x: mnist.test.images, y: mnist.test.labels}))
    print("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))
