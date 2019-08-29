


kernel_init = tf.contrib.layers.xavier_initializer_conv2d()
fire_kernel_init = tf.truncated_normal_initializer(
    stddev=0.001, dtype=tf.float32)

bias_init = tf.constant_initializer(0.0)



#conv1
with tf.name_scope('conv1') as scope:
	channels = inputs.get_shape()[3]
	kernel = tf.get_variable(
	        'kernels', [3, 3, int(channels), 64], initializer=kernel_init, dtype= tf.float32, trainable=True)
	biases = tf.get_variable(
	        'biases', [64], initializer=bias_init, dtype= tf.float32, trainable=True)
	conv = tf.nn.conv2d(
	  inputs, kernel, [1, 1, 2, 1], padding='SAME',
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
fire2 = fire_module('fire2', pool1, s1x1=16, e1x1=64, e3x3=64)
#fire3
fire3 = fire_module('fire3', fire2, s1x1=16, e1x1=64, e3x3=64)
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
fire9 = self.fire_module('fire9', fire8, s1x1=64, e1x1=256, e3x3=256)









def fire_module(layer_name, inputs, s1x1, e1x1, e3x3, stddev=0.001,freeze=False):
# for all conv layer of fire module, size and stride value are 1
	with tf.name_scope(layer_name+'/squeeze1x1') as scope:
		#create kernel and biases
		channels = inputs.get_shape()[3]
		kernel = tf.get_variable(
		        'kernels', [1, 1, int(channels), s1x1], initializer=fire_kernel_init, dtype= tf.float32, trainable=True)
		biases = tf.get_variable(
		        'biases', [s1x1], initializer=bias_init, dtype= tf.float32, trainable=True)


		#convolution layer
		conv = tf.nn.conv2d(
		  inputs, kernel, [1, 1, 1, 1], padding='SAME',
		  name='convolution')
		conv_bias = tf.nn.bias_add(conv, biases, name='bias_add')
		sq1x1 = tf.nn.relu(conv_bias, 'relu')


	with tf.name_scope(layer_name+'/expand1x1') as scope:
		#create kernel and biases
		channels = sq1x1.get_shape()[3]
		kernel = tf.get_variable(
		        'kernels', [1, 1, int(channels), e1x1], initializer=fire_kernel_init, dtype= tf.float32, trainable=True)
		biases = tf.get_variable(
		        'biases', [e1x1], initializer=bias_init, dtype= tf.float32, trainable=True)

		#convolution layer
		conv = tf.nn.conv2d(
		  sq1x1, kernel, [1, 1, 1, 1], padding='SAME',
		  name='convolution')
		conv_bias = tf.nn.bias_add(conv, biases, name='bias_add')
		ex1x1 = tf.nn.relu(conv_bias, 'relu')


	with tf.name_scope(layer_name+'/expand3x3') as scope:
		#create kernel and biases
		channels = sq1x1.get_shape()[3]
		kernel = tf.get_variable(
		        'kernels', [3, 3, int(channels), e3x3], initializer=fire_kernel_init, dtype= tf.float32, trainable=True)
		biases = tf.get_variable(
		        'biases', [e3x3], initializer=bias_init, dtype= tf.float32, trainable=True)

		#convolution layer
		conv = tf.nn.conv2d(
		  sq1x1, kernel, [1, 1, 1, 1], padding='SAME',
		  name='convolution')
		conv_bias = tf.nn.bias_add(conv, biases, name='bias_add')
		ex3x3 = tf.nn.relu(conv_bias, 'relu')


    return tf.concat([ex1x1, ex3x3], 3, name=layer_name+'/concat')