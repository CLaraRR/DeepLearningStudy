import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import pylab

mnist = input_data.read_data_sets("./MNIST_data/", one_hot = True)


# 定义正向传播的结构
# 定义占位符
x = tf.placeholder(tf.float32, [None, 784])  # MNIST数据集的维度是28*28=784
y = tf.placeholder(tf.float32, [None, 10])  # 数字0~9，共10个类别
# 定义学习参数
W = tf.Variable(tf.random_normal([784, 10]), tf.float32)
b = tf.Variable(tf.zeros([10]), tf.float32)
# 定义输出节点
pred = tf.nn.softmax(tf.matmul(x, W) + b)  # softmax分类


# 载入模型进行图像类别预测
saver = tf.train.Saver()
savedir = "./model"
with tf.Session() as sess:
    # 当保存的是某次迭代后得到的模型（训练中断保存了检查点）,载入该模型
    ckpt = tf.train.latest_checkpoint(savedir)  # 得到中断时所保存的checkpoint
    if ckpt != None:
        saver.restore(sess, ckpt)  # 利用这次checkpoint载入模型
    
    output = tf.argmax(pred, 1)
    batch_xs, batch_ys = mnist.train.next_batch(2)
    output_class, pred_prob = sess.run([output, pred], feed_dict = {x: batch_xs, y: batch_ys})
    print("Actual class:", tf.argmax(batch_ys, 1).eval())
    print("Prediction class:", output_class)
    print("One-hot probability:", pred_prob)

    im = batch_xs[0]
    im = im.reshape(-1, 28)
    pylab.imshow(im)
    pylab.show()

    im = batch_xs[1]
    im = im.reshape(-1, 28)
    pylab.imshow(im)
    pylab.show()
