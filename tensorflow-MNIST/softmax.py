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
# placeholder for x and y
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# Weight and Bias
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# softmax
#y = tf.nn.softmax(tf.matmul(x, W) + b)
y = tf.matmul(x, W) + b

# cross entropy
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))

## Trainer def
# train step, 0.5 for learning rate
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

### Tester def
#correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

### Initialize session
# initializer
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

### Training
# train for 1000 step, 100 sample for each batch step
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run(feed_dict={x: batch_xs, y_: batch_ys}) # feed data into placeholsers

### Testing
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
