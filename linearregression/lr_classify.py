import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

def generate(sample_size, mean, cov, diff, regression):
    num_classes = 2
    samples_per_class = int(sample_size/2)

    X0 = np.random.multivariate_normal(mean, cov, samples_per_class)
    Y0 = np.zeros(samples_per_class)

    for ci, d in enumerate(diff):
        X1 = np.random.multivariate_normal(mean + d, cov, samples_per_class)
        Y1 = (ci + 1) * np.ones(samples_per_class)

        X0 = np.concatenate((X0, X1))
        Y0 = np.concatenate((Y0, Y1))

    if regression == False:  # one-hot编码，将0转成[10]
        class_ind = [Y == class_number for class_number in range(num_classes)]
        Y = np.asarray(np.hstack(class_ind), dtype = np.float32)
    X, Y = shuffle(X0, Y0)

    return X, Y

# 生成样本集
np.random.seed(10)
input_dim = 2  # 样本特征数
label_dim = 1  # 标签维数
num_classes = 2  # 标签类别数
mean = np.random.randn(num_classes)
cov = np.eye(num_classes)
train_X, train_Y = generate(1000, mean, cov, [3.0], True)
# colors = ['r' if l == 0 else 'b' for l in train_Y]
# plt.scatter(train_X[:, 0], train_X[:, 1], c = colors)
# plt.xlabel("Scaled age (in yrs)")
# plt.ylabel("Tumor size (in cm)")
# plt.show()



# 构建网络结构
X = tf.placeholder(tf.float32, [None, input_dim])
Y = tf.placeholder(tf.float32, [None, label_dim])
# 定义学习参数
W = tf.Variable(tf.random_normal([input_dim, label_dim]), name = "weight")
b = tf.Variable(tf.zeros([label_dim]), name = "bias")

z = tf.matmul(X, W) + b
output = tf.nn.sigmoid(z)
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = z, labels = Y))
error = tf.reduce_mean(tf.square(Y - output))
optimizer = tf.train.AdamOptimizer(0.04).minimize(cost)  # 使用Adam优化

# 训练
training_epochs = 50
batch_size = 25
saver = tf.train.Saver(max_to_keep= 1)
savedir = "./linearregression/model_classify/"
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        total_error = 0
        for i in range(np.int32(len(train_Y)/batch_size)):
            x1 = train_X[i * batch_size : (i + 1) * batch_size, :]
            y1 = np.reshape(train_Y[i * batch_size : (i + 1) * batch_size], [-1, 1])
            tf.reshape(y1, [-1, 1])
            _, lossval, outputval, errorval = sess.run([optimizer, cost, output, error], feed_dict = {X: x1, Y: y1})
            total_error += errorval
        
        print("Epoch:", epoch + 1, "cost=", lossval, "error=", total_error/np.int32(len(train_Y)/batch_size))
        saver.save(sess, savedir + "linearmodel.ckpt", global_step= epoch + 1)
    
    print("Finished!!!")

    #图形显示
    # train_X, train_Y = generate(100, mean, cov, [3.0],True)
    colors = ['r' if l == 0 else 'b' for l in train_Y[:]]
    plt.scatter(  train_X[:,0], train_X[:,1], c=colors)

    x = np.linspace(-1,8,200) 
    y=-x*(sess.run(W)[0]/sess.run(W)[1])-sess.run(b)/sess.run(W)[1]
    plt.plot(x,y, label='Fitted line')
    plt.legend()
    plt.show() 
