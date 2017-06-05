import tensorflow as tf
import cifar
import layers
import resnet20
import resnet32
import vggnet
import googlenet
import cnn

FLAGS = tf.app.flags.FLAGS

################################################################################
# Select model here
if FLAGS.model_type == 'vggnet':
    model = vggnet
elif FLAGS.model_type == 'googlenet':
    model = googlenet
elif FLAGS.model_type == 'resnet20':
    model = resnet20
elif FLAGS.model_type == 'resnet32':
    model = resnet32
elif FLAGS.model_type == 'cnn':
    model = cnn
else:
    raise ValueError('Not supported type of model: %s' % FLAGS.model_type)

MAX_STEPS = model.MAX_STEPS

def variable_scheduler(var_list, pivot_list, gstep, name=None):
    """Schedule variable according to the global step.
       e.g. var_list = [0.1, 0.01, 0.001], pivot_list = [0, 1000, 2000] then
         0    <= gstep < 1000 --> return 0.1
         1000 <= gstep < 2000 --> return 0.01
         2000 <= gstep        --> return 0.001
    Args:
      var_list: List of variables to return.
      pivot_list: List of pivots when to change the variable.
      gstep: Global step (# of batches trained so far).
      name(Optional): Name of the operation.
    """
    assert(len(var_list) == len(pivot_list))
    if len(var_list) == 1:
        return tf.constant(var_list[0])

    def between(x, a, b):
        return tf.logical_and(tf.greater_equal(x, a), tf.less(x, b))

    # This class is necessary to declare constant lambda expressions
    class temp(object):
        def __init__(self, var):
            self.func = lambda: tf.constant(var)

    gstep = tf.to_int32(gstep)
    conds = {}
    for idx in range(len(pivot_list)-1):
        min_val = tf.constant(pivot_list[idx], tf.int32)
        max_val = tf.constant(pivot_list[idx+1], tf.int32)
        conds[between(gstep, min_val, max_val)] = temp(var_list[idx]).func
    return tf.case(conds, default=temp(var_list[-1]).func, exclusive=True, name=name)

def learning_rate(gstep):
    """Learning rate scheduling. Wrapper of variable_scheduler.
    Args:
      gstep: Global step (# of batches trained so far).
    """
    with tf.name_scope('learning_rate'):
        var_list = model.VAR_LIST
        pivot_list = model.PIVOT_LIST
        return variable_scheduler(var_list, pivot_list, gstep)

def get_inputs():
    """Get input data.
    """
    with tf.name_scope('inputs'):
        # placeholders
        is_train = tf.placeholder(tf.bool, name='is_train')
        images = tf.placeholder(tf.float64, name='batch_images')
        labels = tf.placeholder(tf.float64, name='batch_labels')
        tf.add_to_collection('is_train', is_train)
        tf.add_to_collection('batch_images', images)
        tf.add_to_collection('batch_labels', labels)

        # reshape data
        images = tf.cast(images, tf.float32)
        labels = tf.cast(labels, tf.int32)
        images = tf.reshape(images, [FLAGS.batch_size,cifar.DEPTH,cifar.HEIGHT,cifar.WIDTH])
        images = tf.transpose(images, [0,2,3,1])
        return images, labels

def loss(logits_list, labels):
    """Loss function. Select scheme according to FLAGS.loss_type.

    Args:
      logits_list: List of logits calculated from models to ensemble.
      labels: Label input corresponding to the calculated batch.
    """
    with tf.name_scope('loss'):
        # regularization loss
        vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'inference')
        weights = [var for var in vars if var.name.endswith('weights:0')]
        rloss = model.WD_FACTOR * tf.add_n([tf.nn.l2_loss(w) for w in weights])
        total_loss = rloss

        # classification loss
        num_class = 10
        closs_list = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
                      for logits in logits_list]
        if FLAGS.loss_type == 'independent':
            # independent ensemble
            total_loss += tf.add_n([tf.reduce_mean(closs) for closs in closs_list])
        elif FLAGS.loss_type == 'mcl':
            # stochastic MCL (sMCL) [Stefan Lee et. al., 2016] (https://arxiv.org/abs/1606.07839)
            with tf.device('/cpu:0'):
                min_values, min_index = tf.nn.top_k(-tf.transpose(closs_list), FLAGS.k)
            total_loss -= tf.reduce_sum(min_values)/FLAGS.batch_size
        elif FLAGS.loss_type == 'cmcl_v0':
            # CMCL version 0: confident oracle loss with exact gradient
            a = FLAGS.beta
            softmax_list = [tf.clip_by_value(tf.nn.softmax(logits),1e-10, 1.0) for logits in logits_list]
            entropy_list = [-tf.log(num_class+0.)-tf.reduce_mean(tf.log(softmax),1) for softmax in softmax_list]
            loss_list = []
            for m in range(FLAGS.num_model):
                loss_list.append(closs_list[m] + a*tf.add_n(entropy_list[:m]+entropy_list[m+1:]))
            with tf.device('/cpu:0'):
                temp, min_index = tf.nn.top_k(-tf.transpose(loss_list), FLAGS.k)
            min_index = tf.transpose(min_index)

            new_loss = 0
            for m in range(FLAGS.num_model):
                for topk in range(FLAGS.k):
                    condition = tf.equal(min_index[topk], m)
                    new_loss += tf.reduce_sum(tf.where(condition, closs_list[m] - a*entropy_list[m], tf.zeros(closs_list[0].get_shape())))
            new_loss += tf.reduce_sum(a*tf.add_n(entropy_list))
            total_loss += new_loss/FLAGS.batch_size
        elif FLAGS.loss_type == 'cmcl_v1':
            # CMCL version 1: confident oracle loss with stochastic labeling
            a = FLAGS.beta
            softmax_list = [tf.clip_by_value(tf.nn.softmax(logits),1e-10, 1.0) for logits in logits_list]
            entropy_list = [-tf.log(num_class+0.)-tf.reduce_mean(tf.log(softmax),1) for softmax in softmax_list]
            loss_list = []
            for m in range(FLAGS.num_model):
                loss_list.append(closs_list[m] + a*tf.add_n(entropy_list[:m]+entropy_list[m+1:]))
            with tf.device('/cpu:0'):
                temp, min_index = tf.nn.top_k(-tf.transpose(loss_list), FLAGS.k)
            min_index = tf.transpose(min_index)

            random_labels = tf.random_uniform([FLAGS.num_model,FLAGS.batch_size], minval=0, maxval=10, dtype=tf.int32)
            for m in range(FLAGS.num_model):
                total_condition = tf.constant([False]*FLAGS.batch_size, dtype=tf.bool)
                for topk in range(FLAGS.k):
                    condition = tf.equal(min_index[topk], m)
                    total_condition = tf.logical_or(total_condition, condition)
                    if topk == 0:
                        new_labels = tf.where(condition, labels, random_labels[m])
                    else:
                        new_labels = tf.where(condition, labels, new_labels)

                classification_loss = \
                    tf.where(total_condition,
                              tf.constant([1.]*FLAGS.batch_size),
                              tf.constant([a]*FLAGS.batch_size)) * \
                    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_list[m], labels=new_labels)
                total_loss += tf.reduce_mean(classification_loss)
        else:
            raise ValueError('Not supported type of loss: %s' % FLAGS.loss_type)
    return total_loss

def build():
    """Build the whole calculation graph.
    """
    with tf.device('/cpu:0'):
        # global step - total number of batches processed
        global_step = tf.get_variable('global_step', [],
                                      initializer=tf.constant_initializer(0),
                                      trainable=False)
        # learning rate scheduling
        lr = learning_rate(global_step)
        tf.add_to_collection('learning_rate', lr)

    # get input data
    images, labels = get_inputs()
    with tf.device('/gpu:0'):
        # forward pass
        logits_list = model.inference(images)
        total_loss = loss(logits_list, labels)

        # backward pass
        var_grads = model.OPTIMIZER(lr).compute_gradients(total_loss)
        apply_grads = model.OPTIMIZER(lr).apply_gradients(var_grads, global_step=global_step)

        # softmax results
        softmax_list = [tf.nn.softmax(logits) for logits in logits_list]

        # prediction results
        pred_list = [tf.cast(tf.argmax(softmax, 1), tf.int32) for softmax in softmax_list]

        # comparison is 1 if prediction equals to label, else 0.
        comp_list = [tf.cast(tf.equal(labels, pred), tf.float32) for pred in pred_list]

        # error rate results of models
        err_list = [100.*(1.-tf.reduce_mean(comp)) for comp in comp_list]

        # ensemble top-1 error rate
        pred_top1 = tf.cast(tf.argmax(tf.add_n(softmax_list), 1), tf.int32)
        comp_top1 = tf.cast(tf.equal(labels, pred_top1), tf.float32)
        top1_err = 100.*(1.-tf.reduce_mean(comp_top1))

        # oracle error rate
        comp_oracle = tf.minimum(tf.add_n(comp_list), 1.)
        oracle_err = 100.*(1.-tf.reduce_mean(comp_oracle))

    # training operations
    for op in [apply_grads, global_step, lr, total_loss]:
        tf.add_to_collection('train_ops', op)

    # test operations
    for op in [top1_err, oracle_err]+err_list:
        tf.add_to_collection('test_ops', op)
    return
