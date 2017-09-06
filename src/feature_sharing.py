# Author: Kimin Lee (pokaxpoka@gmail.com) and Changho Hwang (ch.hwang@gmail.com)
# GitHub: https://github.com/chhwang/cmcl
# ==============================================================================
import tensorflow as tf

def feature_sharing(features):
    """Feature sharing operation.
    Args:
      features: List of hidden features from models.
    """
    nmodel = len(features)
    with tf.variable_scope('feature_sharing'):
        shape = features[0].get_shape()
        output = [0.]*nmodel
        for from_idx in range(nmodel):
            for to_idx in range(nmodel):
                if from_idx == to_idx:
                    # don't drop hidden features within a model.
                    mask = 1.
                else:
                    # randomly drop features to share with another model.
                    mask = tf.floor(0.7 + tf.random_uniform(shape))
                output[to_idx] += mask * features[from_idx]
        return output
