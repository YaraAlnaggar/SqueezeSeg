# Author: Bichen Wu (bichen@berkeley.edu) 02/20/2017

"""Neural network model base class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

from utils import util
import numpy as np
import tensorflow as tf


def _variable_on_device(name, shape, initializer, trainable=True):
  """Helper to create a Variable.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  # TODO(bichen): fix the hard-coded data type below
  dtype = tf.float32
  if not callable(initializer):
    var = tf.get_variable(name, initializer=initializer, trainable=trainable)
  else:
    var = tf.get_variable(
        name, shape, initializer=initializer, dtype=dtype, trainable=trainable)
  return var

def _variable_with_weight_decay(name, shape, wd, initializer, trainable=True):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  var = _variable_on_device(name, shape, initializer, trainable)
  if wd is not None and trainable:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

class ModelSkeleton:
  """Base class of NN detection models."""
  def __init__(self, mc):
    self.mc = mc
    self.input_channel = 7
    # a scalar tensor in range (0, 1]. Usually set to 0.5 in training phase and
    # 1.0 in evaluation phase
    self.ph_keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    # projected lidar points on a 2D spherical surface
    self.ph_lidar_input = tf.placeholder(
        tf.float32, [mc.BATCH_SIZE, mc.ZENITH_LEVEL, mc.AZIMUTH_LEVEL, self.input_channel],
        name='lidar_input'
    )
    # A tensor where an element is 1 if the corresponding cell contains an
    # valid lidar measurement. Or if the data is missing, then mark it as 0.
    self.ph_lidar_mask = tf.placeholder(
        tf.float32, [mc.BATCH_SIZE, mc.ZENITH_LEVEL, mc.AZIMUTH_LEVEL, 1],
        name='lidar_mask')
    # A tensor where each element contains the class of each pixel
    # self.ph_label = tf.placeholder(
    #     tf.int32, [mc.BATCH_SIZE, mc.ZENITH_LEVEL, mc.AZIMUTH_LEVEL],
    #     name='label')
    # weighted loss for different classes
    # self.ph_loss_weight = tf.placeholder(
    #     tf.float32, [mc.BATCH_SIZE, mc.ZENITH_LEVEL, mc.AZIMUTH_LEVEL],
    #     name='loss_weight')

    # define a FIFOqueue for pre-fetching data
    self.q = tf.FIFOQueue(
        capacity=mc.QUEUE_CAPACITY,
        dtypes=[tf.float32, tf.float32, tf.float32, tf.int32, tf.float32],
        shapes=[[],
                [mc.BATCH_SIZE, mc.ZENITH_LEVEL, mc.AZIMUTH_LEVEL, self.input_channel],
                [mc.BATCH_SIZE, mc.ZENITH_LEVEL, mc.AZIMUTH_LEVEL, 1],
                [mc.BATCH_SIZE, mc.ZENITH_LEVEL, mc.AZIMUTH_LEVEL],
                [mc.BATCH_SIZE, mc.ZENITH_LEVEL, mc.AZIMUTH_LEVEL]]
    )
    self.enqueue_op = self.q.enqueue(
        [self.ph_keep_prob, self.ph_lidar_input, self.ph_lidar_mask])

    self.keep_prob, self.lidar_input, self.lidar_mask = self.q.dequeue()

    # model parameters
    self.model_params = []

    # model size counter
    self.model_size_counter = [] # array of tuple of layer name, parameter size
    # flop counter
    self.flop_counter = [] # array of tuple of layer name, flop number
    # activation counter
    self.activation_counter = [] # array of tuple of layer name, output activations
    self.activation_counter.append(('input', mc.AZIMUTH_LEVEL*mc.ZENITH_LEVEL*3))


  def _add_forward_graph(self):
    """NN architecture specification."""
    raise NotImplementedError



  def _conv_layer(
      self, layer_name, inputs, filters, size, stride, padding='SAME',
      freeze=False, xavier=False, relu=True, stddev=0.001, bias_init_val=0.0):
    """Convolutional layer operation constructor.

    Args:
      layer_name: layer name.
      inputs: input tensor
      filters: number of output filters.
      size: kernel size.
      stride: stride
      padding: 'SAME' or 'VALID'. See tensorflow doc for detailed description.
      freeze: if true, then do not train the parameters in this layer.
      xavier: whether to use xavier weight initializer or not.
      relu: whether to use relu or not.
      stddev: standard deviation used for random weight initializer.
    Returns:
      A convolutional layer operation.
    """

    mc = self.mc
    use_pretrained_param = False
    if mc.LOAD_PRETRAINED_MODEL:
      cw = self.caffemodel_weight
      if layer_name in cw:
        kernel_val = np.transpose(cw[layer_name][0], [2,3,1,0])
        bias_val = cw[layer_name][1]
        # check the shape
        if (kernel_val.shape == 
              (size, size, inputs.get_shape().as_list()[-1], filters)) \
           and (bias_val.shape == (filters, )):
          use_pretrained_param = True
        else:
          print ('Shape of the pretrained parameter of {} does not match, '
              'use randomly initialized parameter'.format(layer_name))
      else:
        print ('Cannot find {} in the pretrained model. Use randomly initialized '
               'parameters'.format(layer_name))

    if mc.DEBUG_MODE:
      print('Input tensor shape to {}: {}'.format(layer_name, inputs.get_shape()))

    with tf.variable_scope(layer_name) as scope:
      channels = inputs.get_shape()[3]

      # re-order the caffe kernel with shape [out, in, h, w] -> tf kernel with
      # shape [h, w, in, out]
      if use_pretrained_param:
        if mc.DEBUG_MODE:
          print ('Using pretrained model for {}'.format(layer_name))
        kernel_init = tf.constant(kernel_val , dtype=tf.float32)
        bias_init = tf.constant(bias_val, dtype=tf.float32)
      elif xavier:
        kernel_init = tf.contrib.layers.xavier_initializer_conv2d()
        bias_init = tf.constant_initializer(bias_init_val)
      else:
        kernel_init = tf.truncated_normal_initializer(
            stddev=stddev, dtype=tf.float32)
        bias_init = tf.constant_initializer(bias_init_val)

      kernel = _variable_with_weight_decay(
          'kernels', shape=[size, size, int(channels), filters],
          wd=mc.WEIGHT_DECAY, initializer=kernel_init, trainable=(not freeze))

      biases = _variable_on_device('biases', [filters], bias_init, 
                                trainable=(not freeze))
      self.model_params += [kernel, biases]

      conv = tf.nn.conv2d(
          inputs, kernel, [1, 1, stride, 1], padding=padding,
          name='convolution')
      conv_bias = tf.nn.bias_add(conv, biases, name='bias_add')
  
      if relu:
        out = tf.nn.relu(conv_bias, 'relu')
      else:
        out = conv_bias

      self.model_size_counter.append(
          (layer_name, (1+size*size*int(channels))*filters)
      )
      out_shape = out.get_shape().as_list()
      num_flops = \
        (1+2*int(channels)*size*size)*filters*out_shape[1]*out_shape[2]
      if relu:
        num_flops += 2*filters*out_shape[1]*out_shape[2]
      self.flop_counter.append((layer_name, num_flops))

      self.activation_counter.append(
          (layer_name, out_shape[1]*out_shape[2]*out_shape[3])
      )

      return out


  def _pooling_layer(
      self, layer_name, inputs, size, stride, padding='SAME'):
    """Pooling layer operation constructor.

    Args:
      layer_name: layer name.
      inputs: input tensor
      size: kernel size.
      stride: stride
      padding: 'SAME' or 'VALID'. See tensorflow doc for detailed description.
    Returns:
      A pooling layer operation.
    """

    with tf.variable_scope(layer_name) as scope:
      out =  tf.nn.max_pool(inputs, 
                            ksize=[1, size, size, 1], 
                            strides=[1, 1, stride, 1],
                            padding=padding)
      activation_size = np.prod(out.get_shape().as_list()[1:])
      self.activation_counter.append((layer_name, activation_size))
      return out

 

  def _activation_summary(self, x, layer_name):
    """Helper to create summaries for activations.

    Args:
      x: layer output tensor
      layer_name: name of the layer
    Returns:
      nothing
    """
    with tf.variable_scope('activation_summary') as scope:
      tf.summary.histogram(layer_name, x)
      tf.summary.scalar(layer_name+'/sparsity', tf.nn.zero_fraction(x))
      tf.summary.scalar(layer_name+'/average', tf.reduce_mean(x))
      tf.summary.scalar(layer_name+'/max', tf.reduce_max(x))
      tf.summary.scalar(layer_name+'/min', tf.reduce_min(x))
