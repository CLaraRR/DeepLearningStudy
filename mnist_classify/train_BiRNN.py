"""
1.single bidirectional dynamic rnn
2.single bidirectional static rnn

"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 使用one-hot编码的标签
mnist = input_data.read_data_sets("./MNIST_data/", one_hot = True)

n_inputs = 28  # mnist data输入shape:28*28
n_steps = 28  # 序列个数
n_hidden = 128  # 隐藏层个数
n_classes = 10  # mnist分类个数

# 定义占位符
X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
Y = tf.placeholder(tf.float32, [None, n_classes])

x1 = tf.unstack(X, n_steps, 1)


lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias = 1.0)  # forget gate的bias初始为1.0
lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias = 1.0)

# 1.bidirectional dynamic rnn
# outputs, states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, X, dtype = tf.float32)
# outputs = tf.concat(outputs, 2)
# outputs = tf.transpose(outputs, [1, 0, 2])


# 2.bidirectional static rnn
outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x1, dtype = tf.float32)




pred = tf.contrib.layers.fully_connected(outputs[-1], n_classes, activation_fn = None)  # outputs[-1]为时间序列的最后一个输出，shape:(batch_size, ...)


# 定义反向传播的结构
# 使用softmax + cross entropy
cost = tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = Y)
# cost = -tf.reduce_sum(y * tf.log(y_conv))

learning_rate = 1e-3
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))  # tf.argmax返回onehot编码中数值为1的那个元素的下标，也就是类别，tf.equal函数判断两个数是否相等,若相等则返回1，否则返回0
# 计算准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# 训练模型
init = tf.global_variables_initializer()
# 定义参数
training_epochs = 100000  # 迭代次数
batch_size = 128  # 训练过程中一次取多少条数据进行训练，一批次多少条数据
display_step = 10  # 没训练一次就把具体的中间状态显示出来
saver = tf.train.Saver(max_to_keep= 1)
savedir = "./model_lstm/"
with tf.Session() as sess:
    sess.run(init)
    # 启动循环开始训练
    epoch = 0
    while (epoch * batch_size) < training_epochs:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)  # 获得下一个批次的训练数据,数据是随机的
        batch_xs = batch_xs.reshape((batch_size, n_steps, n_inputs))
        # 运行优化器
        sess.run([optimizer], feed_dict = {X: batch_xs, Y: batch_ys})
        # 显示每一次训练完后的详细信息
        if (epoch + 1) % display_step == 0:
            training_accuracy = accuracy.eval(feed_dict = {X: batch_xs, Y: batch_ys}, session = sess)
            print("step %d,accuracy:%g" % (epoch, training_accuracy))
        # saver.save(sess, savedir + "lstm_mnist.ckpt", global_step = epoch + 1)
        epoch += 1
    print("Finished!!!")
    # 测试模型
    # 计算准确率 for 128 mnist test images
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_inputs))
    test_label = mnist.test.labels[:test_len]
    print ("test accuracy: %g" % accuracy.eval(feed_dict = {X: test_data, Y: test_label}))
