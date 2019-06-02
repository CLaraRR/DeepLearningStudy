import tensorflow as tf 
tf.reset_default_graph()
global_step = tf.train.get_or_create_global_step()
step = tf.assign_add(global_step, 1)

with tf.train.MonitoredTrainingSession(checkpoint_dir = "log2/checkpoints", save_checkpoint_secs = 2) as sess:
    print("global step = ", sess.run(global_step))
    for epoch in range(1, 500):
        i = sess.run(step)
        print(i)