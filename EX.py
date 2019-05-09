# -*- coding: utf-8 -*-
"""
Created on Thu May  9 22:11:29 2019
@author: 莫煩
@出處: https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/2-2-example2/
Modified by galileoshen
"""
import tensorflow as tf
import numpy as np

# Create Data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 + 0.3
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights*x_data + biases
loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.3)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()  
sess = tf.Session()
sess.run(init)          # Very Important

print('init', sess.run(Weights), sess.run(biases))

for step in range(401):
    sess.run(train)
    
    print(step, sess.run(Weights), sess.run(biases))
