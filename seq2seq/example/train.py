'''
通过sin与cos进行叠加变形生成无规律的模拟曲线，使用seq2seq模式对其进行学习，
拟合特征，从而达到可以预测下一时刻数据的效果
'''
import random
import math

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def do_generate_x_y(isTrain, batch_size, seqlen):
    batch_x = []
    batch_y = []
    for _ in range(batch_size):
        offset_rand = random.random() * 2 * math.pi
        freq_rand = (random.random() - 0.5) / 1.5 * 15 + 0.5
        amp_rand = random.random() + 0.1

        sin_data = amp_rand * np.sin(np.linspace(
            seqlen / 15.0 * freq_rand * 0.0 * math.pi + offset_rand,
            seqlen / 15.0 * freq_rand * 3.0 * math.pi + offset_rand,
            seqlen * 2
        ))

        offset_rand = random.random() * 2 * math.pi
        freq_rand = (random.random() - 0.5) / 1.5 * 15 + 0.5
        amp_rand = random.random() + 1.2

        sig_data = amp_rand * np.cos(np.linspace(
            seqlen / 15.0 * freq_rand * 0.0 * math.pi + offset_rand,
            seqlen / 15.0 * freq_rand * 3.0 * math.pi + offset_rand,
            seqlen * 2
        )) + sin_data

        batch_x.append(np.array([sig_data[:seqlen]]).T)
        batch_y.append(np.array([sig_data[seqlen:]]).T)

    batch_x = np.array(batch_x).transpose((1, 0, 2))
    batch_y = np.array(batch_y).transpose((1, 0, 2))

    return batch_x, batch_y


# 生成15个连续序列，将cos和sin随机偏移变化后的值叠加起来
def generate_data(isTrain, batch_size):
    seq_length = 15
    if isTrain:
        return do_generate_x_y(isTrain, batch_size, seq_length)
    else:
        return do_generate_x_y(isTrain, batch_size, seq_length * 2)


sample_now, sample_f = generate_data(isTrain=True, batch_size=3)
print('training examples:')
print(sample_now.shape)
# print(sample_now)
# print(sample_f)
print('(seq_length, batch_size, output_dim)')


######## 定义参数及网络结构 #######

seq_length = sample_now.shape[0]
batch_size = 10

output_dim = input_dim = sample_now.shape[-1]
hidden_dim = 12
layers_num = 2

learning_rate = 0.04
nb_iters = 100

lambda_l2_reg = 0.003 # L2正则参数

tf.reset_default_graph()


encoder_input =[]
expected_output = []
decode_input = []

for i in range(seq_length):
    encoder_input.append(tf.placeholder(tf.float32, shape = (None, input_dim)))
    expected_output.append(tf.placeholder(tf.float32, shape=(None, output_dim)))
    decode_input.append(tf.placeholder(tf.float32, shape=(None, input_dim)))

tcells = []
for i in range(layers_num):
    tcells.append(tf.contrib.rnn.GRUCell(hidden_dim))

Mcell = tf.contrib.rnn.MultiRNNCell(tcells)

dec_outputs, dec_memory = tf.contrib.legacy_seq2seq.basic_rnn_seq2seq(encoder_input, decode_input, Mcell)

reshaped_outputs = []
for i in dec_outputs:
    reshaped_outputs.append(tf.contrib.layers.fully_connected(i, output_dim, activation_fn = None))



####### 定义loss和optimizer #######
# 求预测结果和真实结果的loss
output_loss = 0
for _y, _Y in zip(reshaped_outputs, expected_output):
    output_loss += tf.reduce_mean(tf.pow(_y - _Y, 2))

# 求正则化项的loss
reg_loss = 0
for tf_var in tf.trainable_variables():
    if not ('fully_connected' in tf_var.name):
        reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))

loss = output_loss + lambda_l2_reg * reg_loss # 最终的loss等于损失函数和正则化项相加
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)



####### 训练模型 #######
sess = tf.InteractiveSession()

def train_batch(batch_size):
   x, y = generate_data(isTrain=True, batch_size=batch_size)
   feed_dict = {encoder_input[t]: x[t] for t in range(len(encoder_input))}
   feed_dict.update({expected_output[t]: y[t] for t in range(len(expected_output))})

   c = np.concatenate(([np.zeros_like(y[0])], y[:-1]), axis=0)

   feed_dict.update({decode_input[t]: c[t] for t in range(len(c))})

   _, loss_t = sess. run([optimizer, loss], feed_dict=feed_dict)
   return loss_t

def test_batch(batch_size):
    x, y = generate_data(isTrain=True, batch_size=batch_size)
    feed_dict = {encoder_input[t]: x[t] for t in range(len(encoder_input))}
    feed_dict.update({expected_output[t]: y[t] for t in range(len(expected_output))})

    c = np.concatenate(([np.zeros_like(y[0])], y[:-1]), axis=0)

    feed_dict.update({decode_input[t]: c[t] for t in range(len(c))})

    output_lossv, reg_lossv, loss_t = sess.run([output_loss, reg_loss, loss], feed_dict=feed_dict)

    print('output_lossv = {0}\treg_lossv = {1}\tloss_t = {2}'.format(output_lossv, reg_lossv, loss_t))

    return loss_t

train_losses = []
test_losses = []

train_loss = 0.0
test_loss = 0.0

sess.run(tf.global_variables_initializer())
for t in range(nb_iters + 1):
    train_loss = train_batch(batch_size)
    train_losses.append(train_loss)
    if t % 50 == 0:
        test_loss = test_batch(batch_size)
        test_losses.append(test_loss)
        print('step: ', t)
        print('train loss: {0}\ttest loss: {1}'.format(train_loss, test_loss))

print('finished!!!')
print('train loss: {0}\ttest loss: {1}'.format(train_loss, test_loss))

# 输出loss图例
plt.figure(figsize=(12, 6))
plt.plot(np.array(range(0, len(test_losses)))/
         float(len(test_losses) - 1) * (len(train_losses) - 1),
         np.log(test_losses), label = 'Test loss')
plt.plot(np.log(train_losses), label = 'Train loss')
plt.title('Training errors over time (on a logarithmic scale)')
plt.xlabel('Iteration')
plt.ylabel('log(Loss)')
plt.legend(loc = 'best')
plt.show()


# 测试
nb_predictions = 4
print('visualize %d predictions data' % nb_predictions)

preout = []
x, y = generate_data(isTrain=False, batch_size=nb_predictions)
print(np.shape(x), np.shape(y))
for tt in range(seq_length):
    feed_dict = {encoder_input[t]: x[t + tt] for t in range(seq_length)}
    feed_dict.update({expected_output[t]: y[t + tt] for t in range(len(expected_output))})
    c = np.concatenate(([np.zeros_like(y[0])], y[tt: seq_length + tt -1]), axis=0)
    feed_dict.update({decode_input[t]: c[t] for t in range(len(c))})
    outputs = np.array(sess.run([reshaped_outputs], feed_dict=feed_dict)[0])
    preout.append(outputs[-1])

print(np.shape(preout))
preout = np.reshape(preout, [seq_length, nb_predictions, output_dim])


for j in range(nb_predictions):
    plt.figure(figsize=(12, 3))

    for k in range(output_dim):
        past = x[:, j, k]
        expected = y[seq_length-1:, j, k]#对应预测值的打印

        pred = preout[:, j, k]

        label1 = "past" if k == 0 else "_nolegend_"
        label2 = "future" if k == 0 else "_nolegend_"
        label3 = "Pred" if k == 0 else "_nolegend_"
        plt.plot(range(len(past)), past, "o--b", label=label1)
        plt.plot(range(len(past), len(expected) + len(past)),
                 expected, "x--b", label=label2)
        plt.plot(range(len(past), len(pred) + len(past)),
                 pred, "o--y", label=label3)

    plt.legend(loc='best')
    plt.title("Predictions vs. future")
    plt.show()





