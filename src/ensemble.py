from __future__ import print_function
from datetime import datetime
import tensorflow as tf
import numpy as np
import cifar
import time, sys, os

tf.app.flags.DEFINE_string('data_dir', './dataset', 'Directoty to store input dataset')
tf.app.flags.DEFINE_integer('batch_size', 128, 'Number of images to process in a batch.')
tf.app.flags.DEFINE_string('model_type', 'resnet', 'Supported: vggnet, googlenet, resnet')
tf.app.flags.DEFINE_integer('num_model', 5, 'How many models to ensemble.')
tf.app.flags.DEFINE_string('loss_type', 'cmcl_v1', 'Supported: independent, mcl, cmcl_v0, cmcl_v1')
tf.app.flags.DEFINE_integer('k', 4, 'Overlap parameter')
tf.app.flags.DEFINE_float('alpha', 0.75, '')
tf.app.flags.DEFINE_boolean('feature_sharing', True, 'Use feature sharing if True.')
tf.app.flags.DEFINE_boolean('test', True, 'Run test if True else run train')

FLAGS = tf.app.flags.FLAGS

import model

# Set GPU to use. Only one GPU supported.
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def run_train(sess):
    """Train the model.
    Args:
      sess: TensorFlow session to run the model.
    """
    # get data
    train_images, train_labels = cifar.inputs(FLAGS.data_dir, test=False)
    test_images, test_labels = cifar.inputs(FLAGS.data_dir, test=True)

    # shuffle train data
    train_images, train_labels = cifar.shuffle(train_images, train_labels)

    # period of evaluation
    eval_period = cifar.TRAIN_SIZE // FLAGS.batch_size

    # get placeholders
    is_train = tf.get_collection('is_train')[0]
    batch_images = tf.get_collection('batch_images')[0]
    batch_labels = tf.get_collection('batch_labels')[0]

    # record the time when training starts
    start_time = time.time()
    curr_time = start_time
    epoch = 0
    max_test_step = cifar.TEST_SIZE // FLAGS.batch_size

    # loop through training steps
    train_idx = np.array(range(FLAGS.batch_size))
    for step in xrange(model.MAX_STEPS):
        # range of the next train data
        train_idx[train_idx >= cifar.TRAIN_SIZE] -= cifar.TRAIN_SIZE

        # run training
        _, gstep, lr, loss = sess.run(tf.get_collection('train_ops'),
                                      feed_dict={is_train: True,
                                                 batch_images: train_images[train_idx],
                                                 batch_labels: train_labels[train_idx]})
        train_idx += FLAGS.batch_size

        # periodic evaluation
        if step % eval_period == 0:
            elapsed_time = time.time() - curr_time

            # run evaluation with test dataset
            top1_err_sum = 0
            oracle_err_sum = 0
            err_list_sum = np.zeros((FLAGS.num_model,), dtype=np.float)
            test_idx = np.array(range(FLAGS.batch_size))
            for test_step in range(max_test_step):
                test_idx[test_idx >= cifar.TEST_SIZE] -= cifar.TEST_SIZE
                result = sess.run(tf.get_collection('test_ops'),
                                  feed_dict={is_train: False,
                                             batch_images: test_images[test_idx],
                                             batch_labels: test_labels[test_idx]})
                top1_err_sum += result[0]
                oracle_err_sum += result[1]
                err_list_sum += np.asarray(result[2:], dtype=np.float)
                test_idx += FLAGS.batch_size

            # take average
            top1_err = top1_err_sum / float(max_test_step)
            oracle_err = oracle_err_sum / float(max_test_step)
            err_list = err_list_sum / float(max_test_step)

            # print progress
            sys.stdout.write('[%s(+%.1f min)] '
                             'Epoch %d; LR %f; Loss %.6f; Top-1 %.2f%%, Oracle %.2f%%, '
                             'Model Avg %.2f%% (%.1f ms/step)\n' %
                             (datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                              float(time.time()-start_time)/60.,
                              epoch, lr, loss, top1_err, oracle_err,
                              np.average(err_list), 1000*elapsed_time/eval_period))
            sys.stdout.flush()

            # shuffle train data
            train_images, train_labels = cifar.shuffle(train_images, train_labels)
            epoch += 1
            curr_time = time.time()


def run_test(sess):
    """Test the model read from checkpoint.
    Args:
      sess: TensorFlow session to run the model.
    """
    # get data
    test_images, test_labels = cifar.inputs(FLAGS.data_dir, test=True)

    # get placeholders
    is_train = tf.get_collection('is_train')[0]
    batch_images = tf.get_collection('batch_images')[0]
    batch_labels = tf.get_collection('batch_labels')[0]

    top1_err_sum = 0
    oracle_err_sum = 0
    err_list_sum = np.zeros((FLAGS.num_model,), dtype=np.float)
    test_idx = np.array(range(FLAGS.batch_size))
    max_test_step = cifar.TEST_SIZE // FLAGS.batch_size

    print('Running Test ... ')
    start_time = time.time()
    for test_step in range(max_test_step):
        test_idx[test_idx >= cifar.TEST_SIZE] -= cifar.TEST_SIZE
        result = sess.run(tf.get_collection('test_ops'),
                          feed_dict={is_train: False,
                                     batch_images: test_images[test_idx],
                                     batch_labels: test_labels[test_idx]})
        top1_err_sum += result[0]
        oracle_err_sum += result[1]
        err_list_sum += np.asarray(result[2:], dtype=np.float)
        test_idx += FLAGS.batch_size
    elapsed_time = float(time.time() - start_time)
    print('  Elapsed Time %.2f sec' % elapsed_time)

    # take average
    top1_err = top1_err_sum / float(max_test_step)
    oracle_err = oracle_err_sum / float(max_test_step)
    err_list = err_list_sum / float(max_test_step)

    print('  Top-1 Error: %.2f%%' % top1_err)
    print('  Oracle Error: %.2f%%' % oracle_err)
    print('  Best Model Error: %.2f%%' % min(err_list))
    print('  Worst Model Error: %.2f%%' % max(err_list))
    print('  Average Model Error: %.2f%%' % np.average(err_list))


def main(argv=None):
    """Main function.
    """
    # log directory
    log_dir = './log'

    # prepare for checkpoint
    ckpt_dir = './ckpt'
    ckpt_path = ckpt_dir + '/train_result.ckpt'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    # build model
    model.build()

    # create a local session to run training
    with tf.Session() as sess:
        # log the graph data
        writer = tf.summary.FileWriter(log_dir, sess.graph)

        # ckpt saver
        saver = tf.train.Saver()

        if FLAGS.test:
            if not os.path.exists(ckpt_path):
                raise ValueError('Checkpoint %s does not exist.' % ckpt_path)
            saver.restore(sess, ckpt_path)
            print('Restored variabels from %s.' % ckpt_path)
            run_test(sess)
        else:
            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()
            print('Initialized!')
            run_train(sess)
            # last checkpoint
            saver.save(sess, ckpt_path)
            print('  * Variables are saved: %s *' % ckpt_path)


if __name__ == '__main__':
    tf.app.run()
