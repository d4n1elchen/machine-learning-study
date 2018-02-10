#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 00:50:31 2018

@author: daniel
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

### Preparing data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

### Model def
# placeholder for X
x = tf.placeholder(tf.float32, [None, 784])

# Weight and Bias
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# softmax
y = tf.nn.softmax(tf.matmul(x, W) + b)

# cross entropy
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# train step, 0.5 for learning rate
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

### Initialize session
# initializer
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

### Training
# train for 1000 step, 100 sample for each batch step
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys}) # feed data into placeholsers
    
### Testing
#correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))