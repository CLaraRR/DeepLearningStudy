import tensorflow as tf
import numpy as np


def max_pool_with_argmax(net, stride):
    _, mask = tf.nn.max_pool_with_argmax(net, ksize = [1, stride, stride, 1], strides = [1, stride, stride, 1], padding = 'SAME')
    mask = tf.stop_gradient(mask)
    net = tf.nn.max_pool(net, ksize = [1, stride, stride, 1], strides = [1, stride, stride, 1], padding = 'SAME')
    return net, mask  # 返回池化结果和每个最大值的位置


def unpool(net, mask, stride):
    ksize = [1, stride, stride, 1]
    input_shape = net.get_shape().as_list()
    # 计算new shape
    output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])  # 定义反池化后的shape
    # 计算索引
    one_like_mask = tf.ones_like(mask)
    batch_range = tf.reshape(tf.range(output_shape[0], dtype = tf.int64), shape = [input_shape[0], 1, 1, 1])
    b = one_like_mask * batch_range
    y = mask // (output_shape[2] * output_shape[3])
    x = mask % (output_shape[2] * output_shape[3]) // output_shape[3]
    feature_range = tf.range(output_shape[3], dtype=tf.int64)
    f = one_like_mask * feature_range
    # 转置索引
    updates_size = tf.size(net)
    indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, updates_size]))
    values = tf.reshape(net, [updates_size])
    ret = tf.scatter_nd(indices, values, output_shape)
    return net


img = tf.constant([
        [[0.0,4.0],[0.0,4.0],[0.0,4.0],[0.0,4.0]],
        [[1.0,5.0],[1.0,5.0],[1.0,5.0],[1.0,5.0]],
        [[2.0,6.0],[2.0,6.0],[2.0,6.0],[2.0,6.0]],
        [[3.0,7.0],[3.0,7.0], [3.0,7.0],[3.0,7.0]]
])
img = tf.reshape(img, [1, 4, 4, 2])
pooling2 = tf.nn.max_pool(img, [1, 2, 2, 1], [1, 2, 2, 1], padding = 'SAME')  # 得到最大池化的结果
encode, mask = max_pool_with_argmax(img ,2)  # 得到最大池化的结果和最大值的位置
img2 = unpool(encode, mask, 2)  # 得到反池化的结果
with tf.Session() as sess:
    print("image:")
    image = sess.run(img)

    print(image)
    result = sess.run(pooling2)
    print("pooling2:\n", result)
    result, mask2 = sess.run([encode, mask])
    print("encode:\n", result, mask2)
    result =sess.run(img2)
    print("result:\n", result)


