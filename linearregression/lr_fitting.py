import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt 
import time
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

def moving_average(a, w=10):
    if len(a) < w:
        return a[:]
    return [val if idx < w else sum(a[(idx - w) : idx]) / w for idx, val in enumerate(a)]

# 1.准备数据
# 生成一些数据点，数据点大致遵循y=2x这个规律，在生成的时候加入一些噪声
train_x = np.linspace(-1, 1, 100)
train_y = 2 * train_x + np.random.randn(*train_x.shape)*0.3 # 加入一些噪声

# plt.ion()
# plt.figure(1)
# plt.plot(train_x, train_y, 'ro', label="original data")
# plt.legend()
# plt.draw()
# time.sleep(5)
# # plt.close(1)


# 2.搭建模型
# 正向搭建模型
# 占位符
X = tf.placeholder(tf.float32)  # 后面用sess.run()的时候需要用feed_dict的方式将数据传进占位符
Y = tf.placeholder(tf.float32)
# 模型参数
W = tf.Variable(tf.random_normal([1]), name = "weight") # W被初始化成[-1,1]的随机数（正态分布），形状为一维的数组 32
b = tf.Variable(np.zeros([1],np.float32), name = "bias") # b被初始化成全部为0的一维数组
# 前向结构
z = tf.multiply(X, W) + b # z = W*X + b
tf.summary.histogram("z", z)  # 将预测值以直方图形式显示

# 反向搭建模型
# 反向优化
cost = tf.reduce_mean(tf.square(Y - z))
tf.summary.scalar("loss_fuction", cost) # 将损失值以标量形式显示
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


# 3.训练模型

# 初始化所有变量,所有变量的定义一定都要放在这条语句之前
init = tf.global_variables_initializer()
# 定义参数
training_epochs = 20
saver = tf.train.Saver(max_to_keep= 1)
savedir = "./linearregression/model/"
# 启动session
with tf.Session() as sess:
    sess.run(init)
    merged_summary_op = tf.summary.merge_all()  # 合并所有summary
    summary_writer = tf.summary.FileWriter("./linearregression/log", sess.graph)  # 创建summary_writer用于写文件
    plotdata = {"batchsize":[], "loss":[]} # 存放批次值和损失值
    # 向模型输入数据

    for epoch in range(training_epochs):
        for (x, y) in zip(train_x, train_y):
            sess.run(optimizer, feed_dict = {X: x, Y: y})
            summary_str = sess.run(merged_summary_op, feed_dict = {X: x, Y: y})  # 生成summary
            summary_writer.add_summary(summary_str, epoch)  # 将summary写入文件
        # 显示训练中的详细信息
        loss = sess.run(cost, feed_dict={X:train_x, Y:train_y})
        print("Epoch:", epoch +1, "cost=", loss, "W=", sess.run(W), "b=", sess.run(b))
        if not (loss == "NA"):
            plotdata["batchsize"].append(epoch)
            plotdata["loss"].append(loss)
        # 保存模型，为了避免训练中断而失去当前训练出来的参数，所以设置每次迭代保存一次模型，新的会覆盖旧的
        saver.save(sess, savedir + "linearmodel.ckpt", global_step= epoch + 1)
    
    # saver.save(sess, savedir + "linearmodel.ckpt")  # 训练结束后保存最后得到的模型
    print("Finished!")
    # 训练模型可视化
    # 显示原始数据散点图和拟合的线段
    plt.figure(1)
    plt.subplot(211)
    plt.plot(train_x, train_y, "ro", label = "original data")
    plt.plot(train_x, sess.run(W) * train_x + sess.run(b), label = "fitted line")
    plt.title("original data and fitted line")
    plt.legend() # 显示图例

    # 显示每一次训练的损失值曲线图
    plt.subplot(212)
    plotdata["avgloss"] = moving_average(plotdata["loss"])
    plt.plot(plotdata["batchsize"], plotdata["avgloss"], "b--")
    plt.xlabel("minibatch size")
    plt.ylabel("loss")
    plt.title("minibatch run vs. training loss")
    plt.show()

    

# 4.载入模型并使用
saver = tf.train.Saver()
with tf.Session() as sess:
    # 第一种方法，当保存的是训练完后的整个模型
    # saver.restore(sess, savedir + "linearmodel.ckpt")  # 载入模型
    # print("x=0.2, z=", sess.run(z, feed_dict = {X: 0.2}))  # 使用模型

    # 第二种方法，当保存的是训练完后的整个模型
    # saver = tf.train.import_meta_graph(savedir + "linearmodel.ckpt.meta") # 只载入图结构（已经持久化了），不载入图上的运算
    # saver.restore(sess, savedir + "linearmodel.ckpt")
    # print("x=0.2, z=", sess.run(tf.get_default_graph().get_tensor_by_name("add:0"), feed_dict = {X: 0.2}))  # 使用模型
    
    # 第三种方法，当保存的是某次迭代后得到的模型（训练中断保存了检查点）
    ckpt = tf.train.latest_checkpoint(savedir)  # 得到中断时所保存的checkpoint
    if ckpt != None:
        saver.restore(sess, ckpt)  # 利用这次checkpoint载入模型
    print("x=0.2, z=", sess.run(z, feed_dict = {X: 0.2}))  # 使用模型
