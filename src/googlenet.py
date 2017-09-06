# Author: Kimin Lee (pokaxpoka@gmail.com), Changho Hwang (ch.hwang128@gmail.com)
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

    def inception(name, l, wf):
        """Inception module.
        Args:
          name: Scope name of this function.
          l: Output of previous layer.
          wf: Channel width factor of this module.
        """
        with tf.variable_scope(name):
            branchpool = tf.nn.max_pool(l, [1,2,2,1], [1,1,1,1], 'SAME')
            branchpool = layers.conv('conv_pool', branchpool, 32*wf, kernel_size=1)
            branch5x5 = layers.conv('conv_5x5_0', l, 16*wf, kernel_size=1)
            branch5x5 = tf.nn.relu(branch5x5)
            branch5x5 = layers.conv('conv_5x5_1', branch5x5, 32*wf, kernel_size=5)
            branch3x3 = layers.conv('conv_3x3_0', l, 32*wf, kernel_size=1)
            branch3x3 = tf.nn.relu(branch3x3)
            branch3x3 = layers.conv('conv_3x3_1', branch3x3, 64*wf, kernel_size=3)
            branch1x1 = layers.conv('conv_1x1_0', l, 64*wf, kernel_size=1)
            branch1x1 = tf.nn.relu(branch1x1)
            cc = tf.concat([branch1x1,branch3x3,branch5x5,branchpool], 3)
            cc = layers.batchnorm('bn_0', cc, is_train)
            return tf.nn.relu(cc)

    # GoogLeNet-18 inference
    with tf.variable_scope('inference'):
        features = []
        for m in range(FLAGS.num_model):
            l = images
            with tf.variable_scope('model_%d' % m):
                l = layers.conv('conv_init', l, 32, kernel_size=3)
                l = layers.batchnorm('bn_init', l, is_train)
                l = tf.nn.relu(l)
                features.append(l)

        # stochastically share hidden features right before the first pooling
        if FLAGS.feature_sharing:
            features = feature_sharing(features)

        for m in range(FLAGS.num_model):
            l = features[m]
            with tf.variable_scope('model_%d' % m):
                l = tf.nn.max_pool(l, [1,2,2,1], [1,2,2,1], 'VALID')
                l = inception('inception_1a', l, 1)
                l = inception('inception_1b', l, 2)

                l = tf.nn.max_pool(l, [1,2,2,1], [1,2,2,1], 'VALID')
                l = inception('inception_2a', l, 2)
                l = inception('inception_2b', l, 2)
                l = inception('inception_2c', l, 2)
                l = inception('inception_2d', l, 4)

                l = tf.nn.max_pool(l, [1,2,2,1], [1,2,2,1], 'VALID')
                l = inception('inception_3a', l, 4)
                l = inception('inception_3b', l, 4)

                # global average pooling
                l = tf.reduce_mean(l, [1, 2])
                l = layers.fully_connected('fc_0', l, 10)
            features[m] = l
        return features
