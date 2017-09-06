# Author: Kimin Lee (pokaxpoka@gmail.com) and Changho Hwang (ch.hwang@gmail.com)
# GitHub: https://github.com/chhwang/cmcl
# ==============================================================================
import tensorflow as tf
import numpy as np

def conv(name, l, out_channel, kernel_size=3, stride=1, padding='SAME'):
    """Convolution layer.
    
    Args:
      name: Scope name of this function
      l : Output of previous layer
      out_channel: # of channels of each output feature
      kernel_size: Length of a side of convolution filter
      stride: Stride of convolution
      padding: 'SAME' to use padding, or 'VALID'
    """
    in_channel = l.get_shape().as_list()[3]
    with tf.variable_scope(name):
        n = kernel_size * kernel_size * out_channel
        weights = tf.get_variable('weights',
                                  shape=[kernel_size, kernel_size, in_channel, out_channel],
                                  dtype=tf.float32,
                                  initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/n)))
        return tf.nn.conv2d(l, weights, [1, stride, stride, 1], padding=padding)

def fully_connected(name, l, out_dim):
    """Fully connected layer.

    Args:
      name: Scope name of this function
      l : Output of previous layer
      out_dim: Dimension of each output feature
    """
    with tf.variable_scope(name):
        l = tf.reshape(l, [l.get_shape().as_list()[0], -1])
        weights = tf.get_variable('weights', [l.get_shape()[1], out_dim], tf.float32,
                                  initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
        biases = tf.get_variable('biases', [out_dim], tf.float32, initializer=tf.constant_initializer())
        return tf.nn.xw_plus_b(l, weights, biases)

def batchnorm(name, l, is_train):
    """Batch normalization layer.

    Args:
      name: Scope name of this function
      l : Output of previous layer
      is_train: Whether to train or not
    """
    in_channel = l.get_shape().as_list()[3]
    with tf.variable_scope(name):
        beta = tf.get_variable('beta', [in_channel], tf.float32,
                               initializer=tf.constant_initializer(0.0, tf.float32))
        gamma = tf.get_variable('gamma', [in_channel], tf.float32,
                               initializer=tf.constant_initializer(1.0, tf.float32))
        batch_mean, batch_var = tf.nn.moments(l, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.9)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(is_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        return tf.nn.batch_normalization(l, mean, var, beta, gamma, 1e-3)
