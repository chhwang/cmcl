# Author: Kimin Lee (pokaxpoka@gmail.com), Changho Hwang (ch.hwang128@gmail.com)
# GitHub: https://github.com/chhwang/cmcl
# ==============================================================================
from __future__ import print_function

import tensorflow as tf
import numpy as np
import torchfile
import os

TRAIN_SIZE = 73257
TEST_SIZE = 26032
HEIGHT = 32
WIDTH = 32
DEPTH = 3

def inputs(data_dir, test=False):
    """Download and read preprocessed SVHN dataset.
    Args:
      data_dir: Directory to store input dataset.
      test: Read training dataset if False, else read test dataset.
    """
    filepath = os.path.join(data_dir, 'svhn_preprocessed.t7')
    if not os.path.exists(filepath):
        print('Downloading preprocessed SVHN dataset ...')
        data_url = 'https://www.ndsl.kaist.edu/~changho/cmcl/dataset/svhn_preprocessed.t7'
        os.system('mkdir -p %s' % data_dir)
        os.system('wget -P %s %s' % (data_dir, data_url))
        print('Downloaded as: %s' % filepath)
    o = torchfile.load(filepath)
    data = o.testData if test else o.trainData
    images = data.data
    labels = data.labels - 1
    return images, labels

def shuffle(images, labels):
    """Shuffle dataset.
    Args:
      images: List of input images
      labels: List of input labels corresponding to images.
    """
    p = np.random.permutation(len(labels))
    return images[p], labels[p]
