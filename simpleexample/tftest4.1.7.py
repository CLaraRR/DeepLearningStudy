import tensorflow as tf 

a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

add = tf.add(a, b)
mul = tf.multiply(a, b)

with tf.Session() as sess:
    with tf.device("/cpu:0"):  # 指定设备进行运算
        print("3+4=", sess.run(add, feed_dict={a: 3, b: 4}))
        print("3*4=", sess.run(mul, feed_dict={a: 3, b: 4}))