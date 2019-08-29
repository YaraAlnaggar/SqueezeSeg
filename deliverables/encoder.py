# Author: Bichen Wu (bichen@berkeley.edu) 02/20/2017

"""SqueezeSeg model"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import tensorflow as tf



class SqueezeSeg:
  def __init__(self, gpu_id=0, BATCH_SIZE = 1, ZENITH_LEVEL = 64, AZIMUTH_LEVEL = 512, input_channel = 7):
    with tf.device('/gpu:{}'.format(gpu_id)):
      
      self.lidar_input = tf.placeholder(
      tf.float32, [BATCH_SIZE, ZENITH_LEVEL, AZIMUTH_LEVEL, input_channel], name='lidar_input')
      self.kernel_init = tf.contrib.layers.xavier_initializer_conv2d()
      self.fire_kernel_init = tf.truncated_normal_initializer(stddev=0.001, dtype=tf.float32)
      self.bias_init = tf.constant_initializer(0.0)
      self._add_forward_graph()

  def fire_module(self,layer_name, inputs, s1x1, e1x1, e3x3, stddev=0.001,freeze=False):
    # for all conv layer of fire module, size and stride value are 1
    with tf.variable_scope(layer_name+'/squeeze1x1') as scope:
      #create kernel and biases
      channels = inputs.get_shape()[3]
      kernel = tf.get_variable(
              'kernels', [1, 1, int(channels), s1x1], initializer=self.fire_kernel_init, dtype= tf.float32, trainable=True)
      biases = tf.get_variable(
              'biases', [s1x1], initializer=self.bias_init, dtype= tf.float32, trainable=True)


      #convolution layer
      conv = tf.nn.conv2d(
        inputs, kernel, [1, 1, 1, 1], padding='SAME',
        name='convolution')
      conv_bias = tf.nn.bias_add(conv, biases, name='bias_add')
      sq1x1 = tf.nn.relu(conv_bias, 'relu')


    with tf.variable_scope(layer_name+'/expand1x1') as scope:
      #create kernel and biases
      channels = sq1x1.get_shape()[3]
      kernel = tf.get_variable(
              'kernels', [1, 1, int(channels), e1x1], initializer=self.fire_kernel_init, dtype= tf.float32, trainable=True)
      biases = tf.get_variable(
              'biases', [e1x1], initializer=self.bias_init, dtype= tf.float32, trainable=True)

      #convolution layer
      conv = tf.nn.conv2d(
        sq1x1, kernel, [1, 1, 1, 1], padding='SAME',
        name='convolution')
      conv_bias = tf.nn.bias_add(conv, biases, name='bias_add')
      ex1x1 = tf.nn.relu(conv_bias, 'relu')


    with tf.variable_scope(layer_name+'/expand3x3') as scope:
      #create kernel and biases
      channels = sq1x1.get_shape()[3]
      kernel = tf.get_variable(
              'kernels', [3, 3, int(channels), e3x3], initializer=self.fire_kernel_init, dtype= tf.float32, trainable=True)
      biases = tf.get_variable(
              'biases', [e3x3], initializer=self.bias_init, dtype= tf.float32, trainable=True)

      #convolution layer
      conv = tf.nn.conv2d(
        sq1x1, kernel, [1, 1, 1, 1], padding='SAME',
        name='convolution')
      conv_bias = tf.nn.bias_add(conv, biases, name='bias_add')
      ex3x3 = tf.nn.relu(conv_bias, 'relu')


    return tf.concat([ex1x1, ex3x3], 3, name=layer_name+'/concat')

  def _add_forward_graph(self):
    """NN architecture."""
    ## Yara NOTE : fire layer output channel number is the double of the input not same size as in the fire module graph in the paper

    #conv1
    with tf.variable_scope('conv1') as scope:
      channels = self.lidar_input.get_shape()[3]
      kernel = tf.get_variable(
              'kernels', [3, 3, int(channels), 64], initializer=self.kernel_init, dtype= tf.float32, trainable=True)
      biases = tf.get_variable(
              'biases', [64], initializer=self.bias_init, dtype= tf.float32, trainable=True)
      conv = tf.nn.conv2d(
        self.lidar_input, kernel, [1, 1, 2, 1], padding='SAME',
        name='convolution')
      conv_bias = tf.nn.bias_add(conv, biases, name='bias_add')
      conv1 = tf.nn.relu(conv_bias, 'relu')

    #pool1
    with tf.variable_scope('pool1') as scope:
      pool1 =  tf.nn.max_pool(conv1, 
                            ksize=[1, 3, 3, 1], 
                            strides=[1, 1, 2, 1],
                            padding='SAME')
    #fire2
    fire2 = self.fire_module('fire2', pool1, s1x1=16, e1x1=64, e3x3=64)
    #fire3
    fire3 = self.fire_module('fire3', fire2, s1x1=16, e1x1=64, e3x3=64)
    #pool3
    with tf.variable_scope('pool3') as scope:
      pool3 =  tf.nn.max_pool(fire3, 
                            ksize=[1, 3, 3, 1], 
                            strides=[1, 1, 2, 1],
                            padding='SAME')
    #fire4
    fire4 = self.fire_module('fire4', pool3, s1x1=32, e1x1=128, e3x3=128)
    #fire5
    fire5 = self.fire_module('fire5', fire4, s1x1=32, e1x1=128, e3x3=128)
    #pool5
    with tf.variable_scope('pool5') as scope:
      pool5 =  tf.nn.max_pool(fire5, 
                            ksize=[1, 3, 3, 1], 
                            strides=[1, 1, 2, 1],
                            padding='SAME')
    #fire6
    fire6 = self.fire_module('fire6', pool5, s1x1=48, e1x1=192, e3x3=192)
    #fire7
    fire7 = self.fire_module('fire7', fire6, s1x1=48, e1x1=192, e3x3=192)
    #fire8
    fire8 = self.fire_module('fire8', fire7, s1x1=64, e1x1=256, e3x3=256)
    #fire9
    self.output = self.fire_module('fire9', fire8, s1x1=64, e1x1=256, e3x3=256)






lidar_f = np.ones((64,512,7))
model = SqueezeSeg()

session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
init_op = tf.global_variables_initializer()
session.run(init_op)

encoder_out = session.run(
    model.output,
    feed_dict={
        model.lidar_input: [lidar_f]
        }
)

print(encoder_out.shape)

