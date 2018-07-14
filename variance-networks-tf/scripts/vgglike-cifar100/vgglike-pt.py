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


tf.app.flags.DEFINE_string('dataset', 'cifar100', 'dataset name')
tf.app.flags.DEFINE_string('summaries_dir', '', 'global path to summaries directory')
tf.app.flags.DEFINE_string('checkpoints_dir', '', 'global path to checkpoints directory')
tf.app.flags.DEFINE_string('checkpoint', '', 'global path to checkpoint file')
tf.app.flags.DEFINE_string('suffix', '', 'suffix of the logs folder name')
tf.app.flags.DEFINE_integer('batch_size', 100, 'batch size')
tf.app.flags.DEFINE_float('l2', 0.0, 'l2 regularizer coefficient')
FLAGS = tf.app.flags.FLAGS


def conv_bn_rectify(net, num_filters, wd, name, is_training, reuse, pt=False, stochastic=True):
    with tf.variable_scope(name):
        if not pt:
            net = layers.conv_2d_layer(net, [3,3], net.get_shape()[3], num_filters, nonlinearity=None, wd=wd,
                                       padding='SAME', name='conv', with_biases=False)
        else:
            net = layers.pt_conv_2d(net, [3, 3], net.get_shape()[3], num_filters,
                                    padding='SAME', name='conv', with_bias=False, stochastic=stochastic, reuse=reuse)
        biases = layers._variable_on_cpu('biases', net.get_shape()[3], tf.constant_initializer(0.0), dtype=tf.float32)
        net = tf.nn.bias_add(net, biases)
        net = tf.contrib.layers.batch_norm(net, scope=tf.get_variable_scope(), decay=0.9, reuse=reuse, is_training=is_training)
        net = tf.nn.relu(net)
    return net


def net_vgglike(images, nclass, wd, is_training, stohastic, reuse):
    scale = 1
    net = conv_bn_rectify(images, int(64*scale), wd, 'conv_1', is_training, reuse)
    net = tf.contrib.layers.dropout(net, keep_prob=0.7, is_training=stohastic)
    net = conv_bn_rectify(net, int(64*scale), wd, 'conv_2', is_training, reuse)
    net = tf.nn.max_pool(net, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    net = conv_bn_rectify(net, int(128*scale), wd, 'conv_3', is_training, reuse)
    net = tf.contrib.layers.dropout(net, keep_prob=0.6, is_training=stohastic)
    net = conv_bn_rectify(net, int(128*scale), wd, 'conv_4', is_training, reuse)
    net = tf.nn.max_pool(net, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    net = conv_bn_rectify(net, int(256*scale), wd, 'conv_5', is_training, reuse)
    net = tf.contrib.layers.dropout(net, keep_prob=0.6, is_training=stohastic)
    net = conv_bn_rectify(net, int(256*scale), wd, 'conv_6', is_training, reuse)
    net = tf.contrib.layers.dropout(net, keep_prob=0.6, is_training=stohastic)
    net = conv_bn_rectify(net, int(256*scale), wd, 'conv_7', is_training, reuse)
    net = tf.nn.max_pool(net, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    net = conv_bn_rectify(net, int(512*scale), wd, 'conv_8', is_training, reuse)
    net = tf.contrib.layers.dropout(net, keep_prob=0.6, is_training=stohastic)
    net = conv_bn_rectify(net, int(512*scale), wd, 'conv_9', is_training, reuse)
    net = tf.contrib.layers.dropout(net, keep_prob=0.6, is_training=stohastic)
    net = conv_bn_rectify(net, int(512*scale), wd, 'conv_10', is_training, reuse)
    net = tf.nn.max_pool(net, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    net = conv_bn_rectify(net, int(512*scale), wd, 'conv_11', is_training, reuse, pt=True, stochastic=stohastic)
    net = conv_bn_rectify(net, int(512*scale), wd, 'conv_12', is_training, reuse, pt=True, stochastic=stohastic)
    net = conv_bn_rectify(net, int(512*scale), wd, 'conv_13', is_training, reuse, pt=True, stochastic=stohastic)
    net = tf.nn.max_pool(net, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    net = tf.reshape(net, [-1, (net.get_shape()[1]*net.get_shape()[2]*net.get_shape()[3]).value])

    net = tf.contrib.layers.dropout(net, keep_prob=0.5, is_training=stohastic)
    # net = layers.dense_layer(net, net.get_shape()[1].value, 512, name='dense_1', nonlinearity=None, wd=wd)
    net = layers.pt_dense(net, net.get_shape()[1].value, 512, name='dense_1', reuse=reuse, stochastic=stohastic)
    net = tf.contrib.layers.batch_norm(net, scope=tf.get_variable_scope(), decay=0.9, reuse=reuse, is_training=is_training)
    net = tf.nn.relu(net)
    net = tf.contrib.layers.dropout(net, keep_prob=0.5, is_training=stohastic)
    net = layers.dense_layer(net, net.get_shape()[1], nclass, nonlinearity=None, wd=wd, name='dense_2')
    return net


def main(_):
    batch_size = FLAGS.batch_size
    summaries_dir = FLAGS.summaries_dir
    if summaries_dir == '':
        summaries_dir = './logs/vgg_pt_{}_{}'.format(FLAGS.dataset, FLAGS.suffix)
        summaries_dir += time.strftime('_%d-%m-%Y_%H:%M:%S')
    checkpoints_dir = FLAGS.checkpoints_dir
    if checkpoints_dir == '':
        checkpoints_dir = './checkpoints/vgg_pt_{}_{}'.format(FLAGS.dataset, FLAGS.suffix)
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
        inference = lambda x, reuse, is_training, stohastic: net_vgglike(x, nclass, wd, is_training, stohastic, reuse)
        loss = lambda logits, y: metrics.sgvlb(logits, y, len_train)
        train_op, probs_train, probs_test_det, probs_test_stoh, train_loss = utils.build_graph(images, labels, loss, inference, lr, global_step)
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
        n_epochs = 550
        ensemble_size = 5
        lr_policy = lambda epoch_num: policies.linear_decay(
            epoch_num, decay_start=0, total_epochs=n_epochs, start_value=1e-3)
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
            la = tf.get_collection('log_alpha', scope=None)
            print la
            for epoch_num in range(n_epochs):
                train_acc = 0.0
                if epoch_num > 500:
                    ensemble_size = 10

                if epoch_num > 540:
                    ensemble_size = 100

                train_loss_ = 0

                for batch_images, batch_labels in reader.batch_iterator_train_crop_flip(X_train, y_train, batch_size):
                    _, train_probs, summary, train_lossb = sess.run([train_op, probs_train, train_summaries, train_loss],
                                                 feed_dict={lr: lr_policy(epoch_num),
                                                            images: batch_images,
                                                            labels: batch_labels})
                    train_writer.add_summary(summary, global_step.eval())
                    train_loss_ += train_lossb/steps_per_train
                    train_acc += metrics.accurracy_np(train_probs, batch_labels)/steps_per_train
                test_acc_det, test_acc_stoch, test_acc_ens = 0.0, 0.0, 0.0
                for i in range(steps_per_test):
                    batch_images = X_test[i*batch_size:(i+1)*batch_size]
                    batch_labels = y_test[i*batch_size:(i+1)*batch_size]

                    test_probs_stoch = np.zeros([batch_size, nclass])
                    test_probs_det = np.zeros([batch_size, nclass])
                    test_probs_ens = np.zeros([batch_size, nclass])
                    for sample_num in range(ensemble_size):
                        probs_batch_stoch = sess.run([probs_test_stoh], feed_dict={images: batch_images,
                                                                               labels: batch_labels})[0]
                        test_probs_ens += probs_batch_stoch/ensemble_size
                        if sample_num == 0:
                            test_probs_det, la_values = sess.run([probs_test_det, la], feed_dict={images: batch_images, labels: batch_labels})
                            test_probs_stoch = probs_batch_stoch
                    test_acc_det += metrics.accurracy_np(test_probs_det, batch_labels)/steps_per_test
                    test_acc_stoch += metrics.accurracy_np(test_probs_stoch, batch_labels)/steps_per_test
                    test_acc_ens += metrics.accurracy_np(test_probs_ens, batch_labels)/steps_per_test
                saver.save(sess, checkpoints_dir + 'cifar100/cur_model.ckpt')

                epoch_time, start_time = int(time.time() - start_time), time.time()

                print 'epoch_num %3d' % epoch_num,
                print 'train_loss %.3f' % train_loss_,
                print 'train_acc %.3f' % train_acc,
                print 'test_acc_det %.3f' % test_acc_det,
                print 'test_acc_stoch %.3f' % test_acc_stoch,
                print 'test_acc_ens %.3f' % test_acc_ens,
                print 'epoch_time %.3f' % epoch_time,
                print 'la_values', la_values


if __name__ == '__main__':
    tf.app.run()