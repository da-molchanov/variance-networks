import os
import sys
import time
import numpy as np
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


def lenet5_pt(images, nclass, stochastic, reuse):
    # conv 1
    net = layers.pt_conv_2d(images, [5,5], images.get_shape()[3].value, 20, padding='SAME', name='conv_1', reuse=reuse,
                            stochastic=stochastic)
    net = tf.nn.relu(net)
    # max pool
    net = tf.nn.max_pool(net, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    # conv 2
    net = layers.pt_conv_2d(net, [5,5], net.get_shape()[3].value, 50, padding='SAME', name='conv_2', reuse=reuse,
                            stochastic=stochastic)
    net = tf.nn.relu(net)
    # max pool
    net = tf.nn.max_pool(net, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    # reshape
    net = tf.reshape(net, [-1, (net.get_shape()[1]*net.get_shape()[2]*net.get_shape()[3]).value])
    # dense 1
    net = layers.pt_dense(net, net.get_shape()[1].value, 500, name='dense_1', reuse=reuse, stochastic=stochastic)
    net = tf.nn.relu(net)
    # dense 2
    net = layers.pt_dense(net, net.get_shape()[1].value, nclass, name='dense_2', reuse=reuse, stochastic=stochastic)
    return net + 1e-6


def main(_):
    batch_size = FLAGS.batch_size
    summaries_dir = FLAGS.summaries_dir
    if summaries_dir == '':
        summaries_dir = './logs/lenet5_pt_{}_{}'.format(FLAGS.dataset, FLAGS.suffix)
        summaries_dir += time.strftime('_%d-%m-%Y_%H:%M:%S')
    checkpoints_dir = FLAGS.checkpoints_dir
    if checkpoints_dir == '':
        checkpoints_dir = './checkpoints/lenet5_pt_{}_{}'.format(FLAGS.dataset, FLAGS.suffix)
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
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        inference = lambda x, stochastic, reuse: lenet5_pt(x, nclass, stochastic, reuse)
        loss = lambda logits, y: metrics.sgvlb(logits, y, len_train)
        train_op, probs_det, probs_stoch, _ = utils.build_graph_stoch(images, labels, loss, inference, lr, global_step)

        train_summaries = tf.summary.merge_all()

        train_acc_plc = tf.placeholder(tf.float32, shape=[], name='train_acc_stoch_placeholder')
        train_acc_summary = tf.summary.scalar('train_accuracy_stoch', train_acc_plc)
        test_acc_stoch_plc = tf.placeholder(tf.float32, shape=[], name='test_acc_stoch_placeholder')
        test_acc_stoch_summary = tf.summary.scalar('test_accuracy_stoch', test_acc_stoch_plc)
        test_acc_det_plc = tf.placeholder(tf.float32, shape=[], name='test_acc_det_placeholder')
        test_acc_det_summary = tf.summary.scalar('test_accuracy_det', test_acc_det_plc)
        test_acc_ens_plc = tf.placeholder(tf.float32, shape=[], name='test_acc_ens_placeholder')
        test_acc_ens_summary = tf.summary.scalar('test_accuracy_ens', test_acc_ens_plc)
        test_summaries = tf.summary.merge([train_acc_summary, test_acc_stoch_summary, test_acc_det_summary,
                                           test_acc_ens_summary])

        # SUMMARIES WRITERS
        train_writer = tf.summary.FileWriter(summaries_dir + '/train', graph)
        test_writer = tf.summary.FileWriter(summaries_dir + '/test', graph)

        # TRAINING
        n_epochs = 200
        ensemble_size = 10
        lr_policy = lambda epoch_num: policies.linear_decay(epoch_num, decay_start=0,
                                                            total_epochs=n_epochs, start_value=1e-3)
        steps_per_train = len_train/batch_size
        steps_per_test = len_test/batch_size

        saver = tf.train.Saver()
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
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
                    _, train_probs, summary = sess.run([train_op, probs_stoch, train_summaries],
                                                 feed_dict={lr: lr_policy(epoch_num),
                                                            images: batch_images,
                                                            labels: batch_labels})
                    train_writer.add_summary(summary, global_step.eval())
                    train_acc += metrics.accurracy_np(train_probs, batch_labels)/steps_per_train
                test_acc_det, test_acc_stoch, test_acc_ens = 0.0, 0.0, 0.0
                for i in range(steps_per_test):
                    batch_images = X_test[i*batch_size:(i+1)*batch_size]
                    batch_labels = y_test[i*batch_size:(i+1)*batch_size]

                    test_probs_stoch = np.zeros([batch_size, nclass])
                    test_probs_det = np.zeros([batch_size, nclass])
                    test_probs_ens = np.zeros([batch_size, nclass])
                    for sample_num in range(ensemble_size):
                        probs_batch_stoch = sess.run([probs_stoch], feed_dict={images: batch_images,
                                                                               labels: batch_labels})[0]
                        test_probs_ens += probs_batch_stoch/ensemble_size
                        if sample_num == 0:
                            test_probs_det = sess.run([probs_det], feed_dict={images: batch_images,
                                                                              labels: batch_labels})[0]
                            test_probs_stoch = probs_batch_stoch
                    test_acc_det += metrics.accurracy_np(test_probs_det, batch_labels)/steps_per_test
                    test_acc_stoch += metrics.accurracy_np(test_probs_stoch, batch_labels)/steps_per_test
                    test_acc_ens += metrics.accurracy_np(test_probs_ens, batch_labels)/steps_per_test
                saver.save(sess, checkpoints_dir + '/cur_model.ckpt')
                summary = sess.run([test_summaries], feed_dict={test_acc_det_plc: test_acc_det,
                                                                test_acc_stoch_plc: test_acc_stoch,
                                                                test_acc_ens_plc: test_acc_ens,
                                                                train_acc_plc: train_acc})
                for s in summary:
                    test_writer.add_summary(s, global_step.eval())

                epoch_time, start_time = int(time.time() - start_time), time.time()

                print 'epoch_num %3d' % epoch_num,
                print 'train_acc %.3f' % train_acc,
                print 'test_acc_det %.3f' % test_acc_det,
                print 'test_acc_stoch %.3f' % test_acc_stoch,
                print 'test_acc_ens %.3f' % test_acc_ens,
                print 'epoch_time %.3f' % epoch_time


if __name__ == '__main__':
    tf.app.run()
