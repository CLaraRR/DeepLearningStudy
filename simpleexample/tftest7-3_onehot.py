"""
对异或数据进行分类，测试不同激活函数的效果
激活函数：sigmoid, softmax, tanh, relu, leaky relu
标签编码：one-hot
"""
import tensorflow as tf 
import numpy as np

# 生成数据,数据呈异或关系
train_X = [[0, 0], [0, 1], [1, 0], [1, 1]]
train_Y = [[1, 0], [0, 1], [0, 1], [1, 0]]
train_X = np.array(train_X).astype("float32")
train_Y = np.array(train_Y).astype("int16")

# 定义网络模型
learning_rate = 1e-4
n_input = 2
n_label = 2
n_hidden = 100

X = tf.placeholder(tf.float32, [None, n_input])
Y = tf.placeholder(tf.float32, [None, n_label])

weights = {
    "h1": tf.Variable(tf.truncated_normal([n_input, n_hidden], stddev = 0.1)),
    "h2": tf.Variable(tf.truncated_normal([n_hidden, n_label], stddev = 0.1))
}

biases = {
    "h1": tf.Variable(tf.zeros([n_hidden])),
    "h2": tf.Variable(tf.zeros([n_label]))
}

z1 = tf.matmul(X, weights["h1"]) + biases["h1"]
layer_1 = tf.nn.relu(z1)  # 隐层用relu

z2 = tf.matmul(layer_1, weights["h2"]) + biases["h2"]

# 1.loss:sigmoid + 均值平方差
# y_pred = tf.nn.sigmoid(z2)  # 输出层 sigmoid
# cost = tf.reduce_mean(tf.square(y_pred - Y))

# 2.loss:leaky relu + 均值平方差
# y_pred = tf.maximum(z2, 0.01*z2)  # 输出层 leaky relu
# cost = tf.reduce_mean(tf.square(y_pred - Y))

# 3.loss:softmax交叉熵
y_pred = tf.nn.softmax(z2)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = z2, labels = Y))


optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)  # 优化器用Adam

# 训练
training_epochs = 10000  # 输出层sigmoid时10000次，用leaky relu时10000次, softmax交叉熵10000次
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for epoch in range(training_epochs):
    sess.run(optimizer, feed_dict = {X: train_X, Y: train_Y})
    
print("Finished!!!")  
outputval = sess.run(y_pred, feed_dict = {X: train_X, Y: train_Y})
print("output=", outputval) 