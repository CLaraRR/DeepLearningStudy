import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
import time
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

# 1.准备数据train_Xtrain_X
# 生成一些数据点，数据点大致遵循y=2x这个规律，在生成的时候加入一些噪声
train_X = np.linspace(-1, 1, 100)
train_Y = 2 * train_X + np.random.randn(*train_X.shape)*0.3 # 加入一些噪声


# 2.配置分布式Tensorflow

# 定义IP和端口
strps_hosts = "localhost:1681"
strworker_hosts = "localhost:1682,localhost:1683"

# 定义角色名称
strjob_name = "ps"
task_index = 0
# 将字符串转成数组
ps_hosts = strps_hosts.split(",")
worker_hosts = strworker_hosts.split(",")
cluster_spec = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
# 创建server
server = tf.train.Server({"ps": ps_hosts, "worker": worker_hosts}, job_name = strjob_name, task_index= task_index)
# ps角色使用join将线程挂起，等待连接
if strjob_name == "ps":
    print("wait")
    server.join()

# 3.创建网络结构，保证每个终端的网络结构都是一样的
with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task: %d" % task_index, cluster=cluster_spec)):
    X = tf.placeholder(tf.float32)
    Y = tf.placeholder(tf.float32)
    # 模型参数
    W = tf.Variable(tf.random_normal([1]), dtype = tf.float32, name = "weight")
    b = tf.Variable(tf.zeros([1]), dtype = tf.float32, name = "bias")

    global_step = tf.train.get_or_create_global_step()  # 获得迭代次数

    # 前向结构
    z = tf.multiply(X, W) + b # z = W*X + b
    tf.summary.histogram("z", z)  # 将预测值以直方图形式显示

    # 反向搭建模型
    # 反向优化
    cost = tf.reduce_mean(tf.square(Y - z))
    tf.summary.scalar("loss_fuction", cost) # 将损失值以标量形式显示
    learning_rate = 0.01
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost, global_step =global_step)
    # 初始化所有变量,所有变量的定义一定都要放在这条语句之前
    init = tf.global_variables_initializer()


# 4.训练模型，每个终端都有自己的训练任务

# 定义参数
training_epochs = 2200
display_step = 2
saver = tf.train.Saver(max_to_keep= 1)
savedir = "./linearregression/model/"
merged_summary_op = tf.summary.merge_all()  # 合并所有summary

# 创建supervisor，管理session
sv = tf.train.Supervisor(
    is_chief = (task_index == 0),  # 0号worker为chief supervisor
    logdir = "log/super/",
    init_op = init,
    summary_op = None,
    saver = saver,
    global_step = global_step,
    save_model_secs = 5
)
# 连接目标角色创建session
with sv.managed_session(server.target) as sess:
    print("sess ok")
    print(global_step.eval(session = sess))

    
    plotdata = {"batchsize":[], "loss":[]} # 存放批次值和损失值

    for epoch in range(global_step.eval(session = sess), training_epochs*len(train_X)):
        for (x, y) in zip(train_X, train_Y):
            sess.run([optimizer, global_step], feed_dict = {X: x, Y: y})
            summary_str = sess.run(merged_summary_op, feed_dict = {X: x, Y: y})  # 生成summary
            sv.summary_computed(sess, summary_str, global_step = epoch)  # 将summary写入文件
        if epoch % display_step == 0:
            loss = sess.run(cost, feed_dict = {X: train_X, Y: train_Y})
            print("Epoch:", epoch +1, "cost=", loss, "W=", sess.run(W), "b=", sess.run(b))
            if not (loss == "NA"):
                plotdata["batchsize"].append(epoch)
                plotdata["loss"].append(loss)
    print("Finished!")
    sv.saver.save(sess, "log/mnist_with_summaries/" + "sv.cpk")

sv.stop()
        
