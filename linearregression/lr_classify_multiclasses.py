import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
from matplotlib.colors import colorConverter, ListedColormap 

# 对于上面的fit可以这么扩展变成动态的
def onehot(y,start,end):
    ohe = OneHotEncoder()
    a = np.linspace(start,end-1,end-start)
    b =np.reshape(a,[-1,1]).astype(np.int32)
    ohe.fit(b)
    c=ohe.transform(y).toarray()  
    return c     

    
def generate(sample_size, num_classes, diff,regression=False):
    np.random.seed(10)
    mean = np.random.randn(2)
    cov = np.eye(2)  
    
    #len(diff)
    samples_per_class = int(sample_size/num_classes)

    X0 = np.random.multivariate_normal(mean, cov, samples_per_class)
    Y0 = np.zeros(samples_per_class)
    
    for ci, d in enumerate(diff):
        X1 = np.random.multivariate_normal(mean+d, cov, samples_per_class)
        Y1 = (ci+1)*np.ones(samples_per_class)
    
        X0 = np.concatenate((X0,X1))
        Y0 = np.concatenate((Y0,Y1))
        #print(X0, Y0)
    
  
    if regression==False: #one-hot  0 into the vector "1 0
        Y0 = np.reshape(Y0,[-1,1])        
        #print(Y0.astype(np.int32))
        Y0 = onehot(Y0.astype(np.int32),0,num_classes)
        #print(Y0)
    X, Y = shuffle(X0, Y0)
    #print(X, Y)
    return X,Y  


# 生成样本集
# np.random.seed(10)
num_classes = 3  # 标签类别数
input_dim = 2  # 样本特征数
label_dim = num_classes  # 标签维数(因为标签是onehot编码，所以标签维数等于标签类别数)
train_X, train_Y = generate(2000, num_classes, [[3.0], [3.0, 0]], False)
# train_Y_not_onehot = [np.argmax(l) for l in train_Y]
# colors = ['r' if l == 0 else 'b' if l == 1 else 'y' for l in train_Y_not_onehot]
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
output = tf.nn.softmax(z)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = z, labels = Y))  # loss用softmax交叉熵计算
output_labels = tf.argmax(output, axis = 1)
input_labels = tf.argmax(Y, axis = 1)
error = tf.count_nonzero(output_labels - input_labels)
optimizer = tf.train.AdamOptimizer(0.04).minimize(cost)  # 使用Adam优化

# 训练
training_epochs = 50
batch_size = 25
saver = tf.train.Saver(max_to_keep= 1)
savedir = "./linearregression/model_classify_multiclasses/"
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        total_error = 0
        for i in range(np.int32(len(train_Y)/batch_size)):
            x1 = train_X[i * batch_size : (i + 1) * batch_size, :]
            y1 = train_Y[i * batch_size : (i + 1) * batch_size, :]
            _, lossval, outputval, errorval = sess.run([optimizer, cost, output, error], feed_dict = {X: x1, Y: y1})
            total_error += errorval/batch_size
        
        print("Epoch:", epoch + 1, "cost=", lossval, "error=", total_error/np.int32(len(train_Y)/batch_size))
        saver.save(sess, savedir + "linearmodel.ckpt", global_step= epoch + 1)
    
    print("Finished!!!")

    #图形显示
    train_X, train_Y = generate(200, num_classes, [[3.0], [3.0, 0]], False)
    train_Y_not_onehot = [np.argmax(l) for l in train_Y]
    colors = ['r' if l == 0 else 'b' if l == 1 else 'y' for l in train_Y_not_onehot]
    plt.scatter(train_X[:, 0], train_X[:, 1], c = colors)
    plt.xlabel("Scaled age (in yrs)")
    plt.ylabel("Tumor size (in cm)")

    x = np.linspace(-1,8,200) 
    y=-x*(sess.run(W)[0][0]/sess.run(W)[1][0])-sess.run(b)[0]/sess.run(W)[1][0]
    plt.plot(x, y, label='first line')

    y=-x*(sess.run(W)[0][1]/sess.run(W)[1][1])-sess.run(b)[1]/sess.run(W)[1][1]
    plt.plot(x, y, label='second line')

    y=-x*(sess.run(W)[0][2]/sess.run(W)[1][2])-sess.run(b)[2]/sess.run(W)[1][2]
    plt.plot(x, y, label='third line')

    plt.legend()
    plt.show() 


    train_X, train_Y = generate(200, num_classes, [[3.0], [3.0, 0]], False)
    train_Y_not_onehot = [np.argmax(l) for l in train_Y]
    colors = ['r' if l == 0 else 'b' if l == 1 else 'y' for l in train_Y_not_onehot]
    plt.scatter(train_X[:, 0], train_X[:, 1], c = colors)
    plt.xlabel("Scaled age (in yrs)")
    plt.ylabel("Tumor size (in cm)")

    nb_of_xs = 200
    xs1 = np.linspace(-1, 8, num = nb_of_xs)
    xs2 = np.linspace(-1, 8, num = nb_of_xs)
    xx, yy = np.meshgrid(xs1, xs2)  # 创建网格
    # 初始化和填充 classification plane
    classification_plane = np.zeros((nb_of_xs, nb_of_xs))
    for i in range(nb_of_xs):
        for j in range(nb_of_xs):
            classification_plane[i, j] = sess.run(output_labels, feed_dict = {X: [[xx[i, j], yy[i, j]]]})

    # 创建colormap用于显示
    cmap = ListedColormap([
        colorConverter.to_rgba('r', alpha = 0.3),
        colorConverter.to_rgba('b', alpha = 0.3),
        colorConverter.to_rgba('y', alpha = 0.3)
    ])
    # 图示各个样本边界
    plt.contourf(xx, yy, classification_plane, cmap = cmap)
    plt.show()