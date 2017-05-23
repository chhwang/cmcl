import tensorflow as tf
import layers
from feature_sharing import feature_sharing

FLAGS = tf.app.flags.FLAGS

MAX_STEPS = 156250
VAR_LIST = [0.01, 0.002, 0.0004, 0.00008]
PIVOT_LIST = [0, 46875, 93750, 125000]
WD_FACTOR = 0.0005
def OPTIMIZER(lr):
    return tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)

def inference(images):
    """Definition of model inference.
    Args:
      images: A batch of images to process. Shape [batch_size,32,32,3]
    """
    is_train=tf.get_collection('is_train')[0]

    def dropout(l):
        """Dropout layer.
        """
        return tf.cond(is_train,
                       lambda: l*tf.floor(0.5 + tf.random_uniform(l.get_shape())),
                       lambda: l*0.5)

    # CNN inference
    with tf.variable_scope('inference'):
        features = []
        for m in range(FLAGS.num_model):
            l = images
            with tf.variable_scope('model_%d' % m):
                l = layers.conv('conv_0', l, 128, kernel_size=5, padding='VALID')
                l = tf.nn.relu(l)
                features.append(l)

        # stochastically share hidden features right before the first pooling
        if FLAGS.feature_sharing:
            features = feature_sharing(features)

        for m in range(FLAGS.num_model):
            l = features[m]
            with tf.variable_scope('model_%d' % m):
                l = dropout(l)
                l = tf.nn.max_pool(l, [1,2,2,1], [1,2,2,1], 'VALID')
                l = layers.conv('conv_1', l, 256, kernel_size=5, padding='VALID')
                l = tf.nn.relu(l)

                l = dropout(l)
                l = tf.nn.max_pool(l, [1,2,2,1], [1,2,2,1], 'VALID')
                l = layers.fully_connected('fc_0', l, 1024)
                l = tf.nn.relu(l)

                l = dropout(l)
                l = layers.fully_connected('fc_1', l, 10)
            features[m] = l
        return features
