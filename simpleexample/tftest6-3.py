import tensorflow as tf
global_step = tf.Variable(0, trainable = False)
initial_learning_rate = 0.1
decay_rate = 0.9
decay_steps = 10
learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, decay_steps, decay_rate)
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
add_global = global_step.assign_add(1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(learning_rate))
    for i in range(20):
        glo, rate = sess.run([add_global, learning_rate])
        print(glo, rate)