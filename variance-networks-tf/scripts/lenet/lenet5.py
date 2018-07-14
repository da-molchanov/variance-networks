import os
import sys
import time
import tensorflow as tf

if 'SOURCE_CODE_PATH' in os.environ:
    sys.path.append(os.environ['SOURCE_CODE_PATH'])
else:
    sys.path.append(os.getcwd())

from data import reader
from nets import layers, metrics, policies, utils


tf.app.flags.DEFINE_string('dataset', 'mnist', 'dataset name')
tf.app.flags.DEFINE_string('summaries_dir', '', 'global path to summaries directory')
tf.app.flags.DEFINE_string('checkpoints_dir', '', 'global path to checkpoints directory')
tf.app.flags.DEFINE_string('checkpoint', '', 'global path to checkpoint file')
tf.app.flags.DEFINE_string('suffix', '', 'suffix of the logs folder name')
tf.app.flags.DEFINE_integer('batch_size', 128, 'batch size')
tf.app.flags.DEFINE_float('l2', 0.0, 'l2 regularizer coefficient')
FLAGS = tf.app.flags.FLAGS


def lenet5(images, nclass, wd, reuse):
    # conv 1
    net = layers.conv_2d_layer(images, [5,5], images.get_shape()[3].value, 20, nonlinearity=tf.nn.relu, wd=wd,
                               padding='SAME', name='conv_1')
    # max pool
    net = tf.nn.max_pool(net, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    # conv 2
    net = layers.conv_2d_layer(net, [5,5], net.get_shape()[3].value, 50, nonlinearity=tf.nn.relu, wd=wd,
                               padding='SAME', name='conv_2')
    # max pool
    net = tf.nn.max_pool(net, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    # reshape
    net = tf.reshape(net, [-1, (net.get_shape()[1]*net.get_shape()[2]*net.get_shape()[3]).value])
    # dense 1
    net = layers.dense_layer(net, net.get_shape()[1].value, 500, nonlinearity=tf.nn.relu, wd=wd, name='dense_1')
    # dense 2
    net = layers.dense_layer(net, net.get_shape()[1].value, nclass, nonlinearity=None, wd=wd, name='dense_2')
    return net


def main(_):
    batch_size = FLAGS.batch_size
    summaries_dir = FLAGS.summaries_dir
    if summaries_dir == '':
        summaries_dir = './logs/lenet5_{}_{}'.format(FLAGS.dataset, FLAGS.suffix)
        summaries_dir += time.strftime('_%d-%m-%Y_%H:%M:%S')
    checkpoints_dir = FLAGS.checkpoints_dir
    if checkpoints_dir == '':
        checkpoints_dir = './checkpoints/lenet5_{}_{}'.format(FLAGS.dataset, FLAGS.suffix)
        checkpoints_dir += time.strftime('_%d-%m-%Y_%H:%M:%S')
    with tf.Graph().as_default() as graph, tf.device('/gpu:0'):
        # LOADING DATA
        data, len_train, len_test, input_shape, nclass = reader.load(FLAGS.dataset)
        X_train, y_train, X_test, y_test = data

        # BUILDING GRAPH
        images = tf.placeholder(tf.float32, shape=[batch_size, input_shape[1], input_shape[2], input_shape[3]],
                                name='images')
        labels = tf.placeholder(tf.int32, shape=[batch_size], name='labels')
        lr = tf.placeholder(tf.float32, shape=[], name='learning_rate')
        wd = tf.placeholder(tf.float32, shape=[], name='weight_decay')
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        inference = lambda x, reuse: lenet5(x, nclass, wd, reuse)
        loss = lambda logits, y: metrics.log_loss(logits, y, len_train)
        train_op, probs = utils.build_graph(images, labels, loss, inference, lr, global_step)
        train_summaries = tf.summary.merge_all()

        train_acc_plc = tf.placeholder(tf.float32, shape=[], name='train_acc_placeholder')
        train_acc_summary = tf.summary.scalar('train_accuracy_stoch', train_acc_plc)
        test_acc_plc = tf.placeholder(tf.float32, shape=[], name='test_acc_placeholder')
        test_acc_summary = tf.summary.scalar('test_accuracy_det', test_acc_plc)
        test_summaries = tf.summary.merge([train_acc_summary, test_acc_summary])

        # SUMMARIES WRITERS
        train_writer = tf.summary.FileWriter(summaries_dir + '/train', graph)
        test_writer = tf.summary.FileWriter(summaries_dir + '/test', graph)

        # TRAINING
        n_epochs = 50
        lr_policy = lambda epoch_num: policies.linear_decay(
            epoch_num, decay_start=0, total_epochs=n_epochs, start_value=1e-3)
        wd_policy = lambda epoch_num: FLAGS.l2
        steps_per_train = len_train/batch_size
        steps_per_test = len_test/batch_size

        saver = tf.train.Saver()
        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=config) as sess:
            # initialize all variables
            sess.run(tf.global_variables_initializer())

            # restore checkpoints if it's provided
            if FLAGS.checkpoint != '':
                restorer = tf.train.Saver(tf.get_collection('variables'))
                restorer.restore(sess, FLAGS.checkpoint)

            start_time = time.time()
            for epoch_num in range(n_epochs):
                train_acc = 0.0

                for i in range(steps_per_train):
                    batch_images, batch_labels = X_train[i*batch_size:(i+1)*batch_size], \
                                                 y_train[i*batch_size:(i+1)*batch_size]
                    _, probs_batch, summary = sess.run([train_op, probs, train_summaries],
                                                       feed_dict={lr: lr_policy(epoch_num),
                                                                  wd: wd_policy(epoch_num),
                                                                  images: batch_images,
                                                                  labels: batch_labels})
                    train_writer.add_summary(summary, global_step.eval())
                    train_acc += metrics.accurracy_np(probs_batch, batch_labels)/steps_per_train

                test_acc = 0.0
                for i in range(steps_per_test):
                    batch_images = X_test[i*batch_size:(i+1)*batch_size]
                    batch_labels = y_test[i*batch_size:(i+1)*batch_size]

                    probs_batch = sess.run([probs], feed_dict={images: batch_images,
                                                               labels: batch_labels})[0]
                    test_acc += metrics.accurracy_np(probs_batch, batch_labels)/steps_per_test

                saver.save(sess, checkpoints_dir + '/cur_model.ckpt')
                summary = sess.run([test_summaries], feed_dict={test_acc_plc: test_acc, train_acc_plc: train_acc})
                for s in summary:
                    test_writer.add_summary(s, global_step.eval())

                epoch_time, start_time = int(time.time() - start_time), time.time()

                print 'epoch_num %3d' % epoch_num,
                print 'train_acc %.3f' % train_acc,
                print 'test_acc %.3f' % test_acc,
                print 'epoch_time %.3f' % epoch_time


if __name__ == '__main__':
    tf.app.run()
