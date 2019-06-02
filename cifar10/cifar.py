import cifar10_input
import tensorflow as tf 
import numpy as np
import pylab 

# get data
batch_size = 128
data_dir = "./cifar10/cifar10"
images_train, labels_train = cifar10_input.inputs(eval_data = False,  batch_size = batch_size)
images_test , labels_test =  cifar10_input.inputs(eval_data = True, batch_size = batch_size)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
tf.train.start_queue_runners()
image_batch, label_batch = sess.run([images_test, labels_test])
print(image_batch[0])
print(label_batch[0])
pylab.imshow(image_batch[0])
pylab.show()