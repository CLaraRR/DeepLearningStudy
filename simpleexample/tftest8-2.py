'''
用卷积操作提取图片轮廓，卷积核是sobel算子
'''
import tensorflow as tf 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

myimg = mpimg.imread('./simpleexample/img.jpg')
plt.imshow(myimg)
plt.axis('off')
plt.show()
print(myimg.shape)

full = np.reshape(myimg, [1, 870, 580, 3])
inputfull = tf.Variable(tf.constant(1.0, shape = [1, 870, 580, 3]))


# 以下两种filter的写法都可以
filter = tf.Variable(tf.constant(
[[-1.0, -1.0, -1.0], [0, 0, 0], [1.0, 1.0, 1.0],
[-2.0, -2.0, -2.0], [0, 0, 0], [2.0, 2.0, 2.0],
[-1.0, -1.0, -1.0], [0, 0, 0], [1.0, 1.0, 1.0]],
 shape = [3, 3, 3, 1])
)

# filter = tf.Variable(tf.constant(
# [-1.0, -1.0, -1.0, 0, 0, 0, 1.0, 1.0, 1.0,
# -2.0, -2.0, -2.0, 0, 0, 0, 2.0, 2.0, 2.0,
# -1.0, -1.0, -1.0, 0, 0, 0, 1.0, 1.0, 1.0],
#  shape = [3, 3, 3, 1])
# )

op = tf.nn.conv2d(inputfull, filter, strides = [1, 1, 1, 1], padding = 'SAME')
o = tf.cast(((op - tf.reduce_min(op))/(tf.reduce_max(op)-tf.reduce_min(op)))*255, tf.uint8)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    t, f = sess.run([o, filter], feed_dict = {inputfull: full})

    t = np.reshape(t, [870, 580])

    plt.imshow(t, cmap = 'Greys_r')
    plt.axis('off')
    plt.show()