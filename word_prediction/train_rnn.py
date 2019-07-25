# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import random
import time
from collections import Counter



start_time = time.time()

def elapsed(sec):
    if sec < 60:
        return str(sec) + 'sec'
    elif sec < (60 * 60):
        return str(sec/60) + 'min'
    else:
        return str(sec/(60*60)) + 'hr'


tf.reset_default_graph()
training_file = 'wordtest.txt'


# 处理多个中文文件
def readalltxt(txt_files):
    labels = []
    for txt_file in txt_files:
        target = get_ch_label(txt_file)
        labels.append(target)

    return labels


# 处理汉字
def get_ch_label(txt_file):
    labels = ''
    with open(txt_file, 'rb') as f:
        for label in f:
            labels = labels + label.decode('utf-8')

    return labels


# 优先转文件里的字符到向量
def get_ch_label_v(txt_file, word_num_map, txt_label=None):
    words_size = len(word_num_map)
    to_num = lambda word: word_num_map.get(word, words_size)
    if txt_file is not None:
        txt_label = get_ch_label(txt_file)

    labels_vector = list(map(to_num, txt_label))

    return labels_vector


####### 样本预处理 #######
training_data = get_ch_label(training_file)
print('loaded training data...')

counter = Counter(training_data) # 统计每个字出现的次数
words = sorted(counter) # 根据每个字出现的频率排序
words_size = len(words)
word_num_map = dict(zip(words, range(words_size))) # 字和序号一一对应

print('字表大小：', words_size)

wordlabel = get_ch_label_v(training_file, word_num_map) # 获取训练文本的label向量


####### 构建模型 #######

# 定义参数
learning_rate = 0.001
training_epochs = 10000
display_step = 1000
n_input = 4

n_hidden1 = 256
n_hidden2 = 512
n_hidden3 = 512

# 定义占位符
x = tf.placeholder(tf.float32, [None, n_input, 1])
y = tf.placeholder(tf.float32, [None, words_size]) # 使用one-hot encoding

# 定义网络结构
x1 = tf.reshape(x, [-1, n_input])
x2 = tf.split(x1, n_input, 1)

stacked_rnn = [rnn.LSTMCell(n_hidden1), rnn.LSTMCell(n_hidden2), rnn.LSTMCell(n_hidden3)]
rnn_cell = rnn.MultiRNNCell(stacked_rnn)

# 通过static_rnn得到输出
outputs, states = rnn.static_rnn(rnn_cell, x2, dtype = tf.float32)

# 通过全连接输出指定维度
pred = tf.contrib.layers.fully_connected(outputs[-1], words_size, activation_fn = None)

# 定义优化器
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# 模型评估
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y ,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


####### 训练模型 #######
init = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep= 1)
savedir = "./model/"

with tf.Session() as session:
    session.run(init)
    epoch = 0
    offset = random.randint(0, n_input + 1)
    end_offset = n_input + 1
    acc_total = 0
    loss_total = 0

    ckpt = tf.train.latest_checkpoint(savedir)
    print('ckpt:', ckpt)
    startepoch = 0

    if ckpt != None:
        saver.restore(session, ckpt)
        ind = ckpt.find('-')
        startepoch = int(ckpt[ind + 1:])
        print(startepoch)
        epoch = startepoch

    while epoch < training_epochs:
        # 随机取一个位置偏移
        if offset > (len(training_data) - end_offset):
            offset = random.randint(0, n_input + 1)

        inwords = [[wordlabel[i]] for i in range(offset, offset + n_input)] # 按照指定的位置偏移获得后4个文字向量，当做输入

        inwords = np.reshape(np.array(inwords), [-1, n_input, 1])

        out_onehot = np.zeros([words_size], dtype = np.float32)
        out_onehot[wordlabel[offset + n_input]] = 1.0
        out_onehot = np.reshape(out_onehot, [1, -1]) # 所有的字都变成onehot

        _, acc, lossval, onehot_pred = session.run([optimizer, accuracy, loss, pred], feed_dict={x: inwords, y: out_onehot})
        loss_total += lossval
        acc_total += acc

        if (epoch + 1) % display_step == 0:
            print('epoch:', epoch)
            print('avg loss=' + '{:.6f}'.format(loss_total/display_step) + ', avg accuracy=' + '{:.2f}'.format(100*acc_total/display_step))
            acc_total = 0
            loss_total = 0
            # 输出训练时序列输入以及预测结果
            in2 = [words [wordlabel[i]] for i in range(offset, offset + n_input)]
            out2 = words [wordlabel[offset + n_input]]
            out_pred = words[int(tf.argmax(onehot_pred, 1).eval())]
            print('%s->label:[%s] vs pred:[%s]'%(in2, out2, out_pred))
            saver.save(session, savedir + 'wordpred_rnn.ckpt', global_step=epoch)

        epoch += 1
        offset += (n_input + 1)

    print('finished!!!')
    saver.save(session, savedir + 'wordpred_rnn.ckpt', global_step=epoch)
    print('elapsed time:', elapsed(time.time() - start_time))




    ####### 根据用户输入生成句子 #######
    while True:
        sentence = input('请输入%s个字：' % n_input)
        inputword = sentence.strip()

        if len(inputword) != n_input:
            print('您输入的字符长度为：', len(inputword), '请输入%s个字！'%n_input)
            continue

        try:
            inputword = get_ch_label_v(None, word_num_map, inputword)
            for i in range(50):
                keys = np.reshape(np.array(inputword), [-1, n_input, 1])
                onehot_pred = session.run(pred, feed_dict={x: keys})
                onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
                sentence = '%s%s' % (sentence, words[onehot_pred_index])
                inputword = inputword[1:]
                inputword.append(onehot_pred_index)

            print(sentence)

        except:
            print('该字我还没学会！')
