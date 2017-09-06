# Author: Kimin Lee (pokaxpoka@gmail.com) and Changho Hwang (ch.hwang@gmail.com)
# GitHub: https://github.com/chhwang/cmcl
# ==============================================================================
import tensorflow as tf
import layers
from feature_sharing import feature_sharing

FLAGS = tf.app.flags.FLAGS

if FLAGS.dataset == 'cifar':
    MAX_STEPS = 39063
    VAR_LIST = [0.1, 0.02, 0.004, 0.0008]
    PIVOT_LIST = [0, 9766, 19532, 29297]
    WD_FACTOR = 0.0005
elif FLAGS.dataset == 'svhn':
    MAX_STEPS = 58200
    VAR_LIST = [0.1, 0.02, 0.004, 0.0008]
    PIVOT_LIST = [0, 14550, 29100, 43650]
    WD_FACTOR = 0.0005
else:
    raise ValueError('Not supported dataset: %s' % FLAGS.dataset)

def OPTIMIZER(lr):
    return tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)

def inference(images):
    """Definition of model inference.
    Args:
      images: A batch of images to process. Shape [batch_size,32,32,3]
    """
    is_train=tf.get_collection('is_train')[0]

    def conv_bn_relu(name, l, out_channel):
        """A sequence of convolution, batch normalization and ReLU.
        Args:
          name: Scope name of this function.
          l: Output of previous layer.
          out_channel: # of channels of each output feature.
        """
        with tf.variable_scope(name):
            l = layers.conv('conv_0', l, out_channel)
            l = layers.batchnorm('bn_0', l, is_train)
            return tf.nn.relu(l)

    # VGGNet-17 inference
    with tf.variable_scope('inference'):
        features = []
        for m in range(FLAGS.num_model):
            l = images
            with tf.variable_scope('model_%d' % m):
                l = conv_bn_relu('conv_bn_relu_01', l, 64)
                l = conv_bn_relu('conv_bn_relu_02', l, 64)
                features.append(l)

        # stochastically share hidden features right before the first pooling
        if FLAGS.feature_sharing:
            features = feature_sharing(features)

        for m in range(FLAGS.num_model):
            l = features[m]
            with tf.variable_scope('model_%d' % m):
                l = tf.nn.max_pool(l, [1,2,2,1], [1,2,2,1], 'VALID')
                l = conv_bn_relu('conv_bn_relu_03', l, 128)
                l = conv_bn_relu('conv_bn_relu_04', l, 128)

                l = tf.nn.max_pool(l, [1,2,2,1], [1,2,2,1], 'VALID')
                l = conv_bn_relu('conv_bn_relu_05', l, 256)
                l = conv_bn_relu('conv_bn_relu_06', l, 256)
                l = conv_bn_relu('conv_bn_relu_07', l, 256)
                l = conv_bn_relu('conv_bn_relu_08', l, 256)

                l = tf.nn.max_pool(l, [1,2,2,1], [1,2,2,1], 'VALID')
                l = conv_bn_relu('conv_bn_relu_09', l, 512)
                l = conv_bn_relu('conv_bn_relu_10', l, 512)
                l = conv_bn_relu('conv_bn_relu_11', l, 512)
                l = conv_bn_relu('conv_bn_relu_12', l, 512)

                l = tf.nn.max_pool(l, [1,2,2,1], [1,2,2,1], 'VALID')
                l = conv_bn_relu('conv_bn_relu_13', l, 512)
                l = conv_bn_relu('conv_bn_relu_14', l, 512)
                l = conv_bn_relu('conv_bn_relu_15', l, 512)
                l = conv_bn_relu('conv_bn_relu_16', l, 512)

                # global average pooling
                l = tf.reduce_mean(l, [1, 2])
                l = layers.fully_connected('fc_0', l, 10)
            features[m] = l
        return features
