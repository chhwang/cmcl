import tensorflow as tf
import layers
from feature_sharing import feature_sharing

FLAGS = tf.app.flags.FLAGS

MAX_STEPS = 64000
VAR_LIST = [0.1, 0.01, 0.001]
PIVOT_LIST = [0, 32000, 48000]
WD_FACTOR = 0.0001
def OPTIMIZER(lr):
    return tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=False)

def inference(images):
    """Definition of model inference.
    Args:
      images: A batch of images to process. Shape [batch_size,32,32,3]
    """
    is_train=tf.get_collection('is_train')[0]

    def shortcut(l, in_channel, out_channel):
        """Shortcut for residual function.
        Args:
          l: Output of previous layer.
          in_channel: # of channels of l.
          out_channel: # of channels of each output feature.
        """
        shortcut = tf.nn.avg_pool(l, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
        pad = (out_channel - in_channel)//2
        return tf.pad(shortcut, [[0,0], [0,0], [0,0], [pad, pad]])

    def residual(name, l, in_channel, out_channel, stride):
        """Residual function.
        Args:
          name: Scope name of this function.
          l: Output of previous layer.
          in_channel: # of channels of l.
          out_channel: # of channels of each output feature.
          stride: Stride of the first convolution in residual function.
        """
        with tf.variable_scope(name):
            sc = l if stride == 1 else shortcut(l, in_channel, out_channel)
            l = layers.conv('conv_0', l, out_channel, stride=stride)
            l = layers.batchnorm('bn_0', l, is_train)
            l = tf.nn.relu(l)
            l = layers.conv('conv_1', l, out_channel, stride=1)
            l = layers.batchnorm('bn_1', l, is_train)
            l = tf.nn.relu(l + sc)
            return l

    # ResNet-20 inference
    with tf.variable_scope('inference'):
        features = []
        for m in range(FLAGS.num_model):
            l = images
            with tf.variable_scope('model_%d' % m):
                l = layers.conv('conv_init', l, 16, stride=1)
                l = residual('res_1_1', l, 16, 16, 1)
                l = residual('res_1_2', l, 16, 16, 1)
                l = residual('res_1_3', l, 16, 16, 1)
                features.append(l)

        # stochastically share hidden features right before the first pooling
        if FLAGS.feature_sharing:
            features = feature_sharing(features)

        for m in range(FLAGS.num_model):
            l = features[m]
            with tf.variable_scope('model_%d' % m):
                l = residual('res_2_1', l, 16, 32, 2)
                l = residual('res_2_2', l, 32, 32, 1)
                l = residual('res_2_3', l, 32, 32, 1)

                l = residual('res_3_1', l, 32, 64, 2)
                l = residual('res_3_2', l, 64, 64, 1)
                l = residual('res_3_3', l, 64, 64, 1)

                l = layers.batchnorm('bn_0', l, is_train)
                l = tf.nn.relu(l)
                # global average pooling
                l = tf.reduce_mean(l, [1, 2])
                l = layers.fully_connected('fc_0', l, 10)
            features[m] = l
        return features
