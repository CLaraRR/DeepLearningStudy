"""
对异或数据进行分类，通过正则化改善过拟合情况的例子
正则化：L2范数
200个神经元
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from matplotlib.colors import colorConverter, ListedColormap 
from sklearn.preprocessing import OneHotEncoder

def onehot(y, start, end):
    ohe = OneHotEncoder()
    a = np.linspace(start, end - 1, end - start)
    b = np.reshape(a, [-1, 1]).astype(np.int32)
    ohe.fit(b)
    c = ohe.transform(y).toarray()  
    return c  
    
def generate(sample_size, num_classes, diff, regression = False):
    np.random.seed(10)
    mean = np.random.randn(2)
    cov = np.eye(2)  
    
    #len(diff)
    samples_per_class = int(sample_size / num_classes)

    X0 = np.random.multivariate_normal(mean, cov, samples_per_class)
    Y0 = np.zeros(samples_per_class)
    
    for ci, d in enumerate(diff):
        X1 = np.random.multivariate_normal(mean + d, cov, samples_per_class)
        Y1 = (ci + 1) * np.ones(samples_per_class)
    
        X0 = np.concatenate((X0, X1))
        Y0 = np.concatenate((Y0, Y1))

  
    if regression == False: #one-hot  0 into the vector "1 0
        Y0 = np.reshape(Y0, [-1, 1])        
        #print(Y0.astype(np.int32))
        Y0 = onehot(Y0.astype(np.int32), 0, num_classes)
        #print(Y0)
    X, Y = shuffle(X0, Y0)
    #print(X, Y)
    return X,Y 

# 生成一些模拟数据
np.random.seed(10)

input_dim = 2
num_classes = 4 
train_X, train_Y = generate(320, num_classes, [[3.0,0],[3.0,3.0],[0,3.0]], True)
train_Y = train_Y % 2 
# xr = []
# xb = []
# for(l, k) in zip(train_Y[:], train_X[:]):
#     if l == 0.0 :
#         xr.append([k[0], k[1]])        
#     else:
#         xb.append([k[0], k[1]])
# xr = np.array(xr)
# xb = np.array(xb)      
# plt.scatter(xr[:, 0], xr[:, 1], c = 'r', marker = '+')
# plt.scatter(xb[:, 0], xb[:, 1], c = 'b', marker = 'o')
# plt.show() 


# 定义网络结构
train_Y = np.reshape(train_Y, [-1, 1])
learning_rate = 1e-4
n_input  = 2
n_label  = 1
#n_hidden = 2  # 2个神经元，欠拟合
n_hidden = 200  # 200个神经元，过拟合

X = tf.placeholder(tf.float32, [None, n_input])
Y = tf.placeholder(tf.float32, [None, n_label])

weights = {
    'h1': tf.Variable(tf.truncated_normal([n_input, n_hidden], stddev = 0.1)),
    'h2': tf.Variable(tf.random_normal([n_hidden, n_label], stddev = 0.1))
	} 
biases = {
    'h1': tf.Variable(tf.zeros([n_hidden])),
    'h2': tf.Variable(tf.zeros([n_label]))
    }    

z1 = tf.matmul(X, weights["h1"]) + biases["h1"]
layer_1 = tf.nn.relu(z1)  # 隐层用relu

z2 = tf.matmul(layer_1, weights["h2"]) + biases["h2"]
y_pred = tf.maximum(z2, 0.01*z2)  # 输出层 leaky relu

reg = 0.01  # 正则项的参数
# loss用均值方差计算,加上L2正则项来防止过拟合
cost = tf.reduce_mean(tf.square(y_pred - Y)) + tf.nn.l2_loss(weights["h1"]) * reg +  tf.nn.l2_loss(weights["h2"]) * reg 
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)  # 优化器用Adam

# 训练
training_epochs = 40000  # 输出层用tanh时10000次，用leaky relu时40000次
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for epoch in range(training_epochs):
    _, lossval = sess.run([optimizer, cost], feed_dict = {X: train_X, Y: train_Y})
    
    if epoch % 1000 == 0:
        print("Epoch:", epoch + 1, "cost=", lossval)

print("Finished!!!")  

# 显示数据和分类效果
xr = []
xb = []
for(l, k) in zip(train_Y[:], train_X[:]):
    if l == 0.0 :
        xr.append([k[0], k[1]])        
    else:
        xb.append([k[0], k[1]])
xr = np.array(xr)
xb = np.array(xb)      
plt.scatter(xr[:, 0], xr[:, 1], c = 'r', marker = '+')
plt.scatter(xb[:, 0], xb[:, 1], c = 'b', marker = 'o')

nb_of_xs = 200
xs1 = np.linspace(-3, 10, num = nb_of_xs)
xs2 = np.linspace(-3, 10, num = nb_of_xs)
xx, yy = np.meshgrid(xs1, xs2) # create the grid
# Initialize and fill the classification plane
classification_plane = np.zeros((nb_of_xs, nb_of_xs))
for i in range(nb_of_xs):
    for j in range(nb_of_xs):
        #classification_plane[i,j] = nn_predict(xx[i,j], yy[i,j])
        classification_plane[i, j] = sess.run(y_pred, feed_dict={X: [[ xx[i, j], yy[i, j] ]]} )
        classification_plane[i, j] = int(classification_plane[i, j])

# Create a color map to show the classification colors of each grid point
cmap = ListedColormap([
        colorConverter.to_rgba('r', alpha = 0.30),
        colorConverter.to_rgba('b', alpha = 0.30)])
# Plot the classification plane with decision boundary and input samples
plt.contourf(xx, yy, classification_plane, cmap = cmap)
plt.show()

################################################################
# 验证模型过拟合了

# 生成一些验证数据点
validate_X, validate_Y = generate(12, num_classes, [[3.0, 0], [3.0, 3.0], [0, 3.0]], True)
validate_Y = validate_Y % 2

xr = []
xb = []
for(l, k) in zip(validate_Y[:], validate_X[:]):
    if l == 0.0 :
        xr.append([k[0], k[1]])        
    else:
        xb.append([k[0], k[1]])
xr = np.array(xr)
xb = np.array(xb)      
plt.scatter(xr[:, 0], xr[:, 1], c = 'r', marker = '+')
plt.scatter(xb[:, 0], xb[:, 1], c = 'b', marker = 'o')

validate_Y = np.reshape(validate_Y, [-1, 1])           
print ("loss:\n", sess.run(cost, feed_dict={X: validate_X, Y: validate_Y}))   

nb_of_xs = 200
xs1 = np.linspace(-1, 8, num = nb_of_xs)
xs2 = np.linspace(-1, 8, num = nb_of_xs)
xx, yy = np.meshgrid(xs1, xs2) # create the grid
# Initialize and fill the classification plane
classification_plane = np.zeros((nb_of_xs, nb_of_xs))
for i in range(nb_of_xs):
    for j in range(nb_of_xs):
        #classification_plane[i,j] = nn_predict(xx[i,j], yy[i,j])
        classification_plane[i, j] = sess.run(y_pred, feed_dict={X: [[xx[i, j], yy[i, j]]]} )
        classification_plane[i, j] = int(classification_plane[i, j])

# Create a color map to show the classification colors of each grid point
cmap = ListedColormap([
        colorConverter.to_rgba('r', alpha = 0.30),
        colorConverter.to_rgba('b', alpha = 0.30)])
# Plot the classification plane with decision boundary and input samples
plt.contourf(xx, yy, classification_plane, cmap=cmap)
plt.show()   
      