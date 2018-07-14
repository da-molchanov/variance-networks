import tensorflow as tf
import numpy as np


def log_loss(logits, labels, num_examples, reuse=False):
    labels = tf.cast(labels, tf.int32)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits,
                                                                   name='cross_entropy_per_example')
    cross_entropy_mean = num_examples * tf.reduce_mean(cross_entropy, name='cross_entropy')
    total_loss = cross_entropy_mean
    if not reuse:
        tf.summary.scalar('nll_loss', cross_entropy_mean)
    return total_loss


def sgvlb(logits, labels, num_examples, reuse=False):
    labels = tf.cast(labels, tf.int32)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits,
                                                                   name='cross_entropy_per_example')
    cross_entropy_mean = num_examples * tf.reduce_mean(cross_entropy, name='cross_entropy')
    kl_loss = 0
    if len(tf.get_collection('kl_loss')) > 0:
        kl_loss = tf.add_n(tf.get_collection('kl_loss'))
    total_loss = cross_entropy_mean + kl_loss
    elbo = cross_entropy_mean + kl_loss
    if not reuse:
        tf.summary.scalar('kl_loss', kl_loss)
        tf.summary.scalar('nll_loss', cross_entropy_mean)
        tf.summary.scalar('total_loss', total_loss)
        tf.summary.scalar('elbo', elbo)
    return total_loss


def accuracy_tf(logits, labels):
    predicted_labels = tf.cast(tf.argmax(logits, axis=1), dtype=tf.int32)
    return tf.reduce_mean(tf.cast(tf.equal(predicted_labels, labels), tf.float32))


def accurracy_np(probs, labels):
    predicted_labels = np.argmax(probs, axis=1)
    return np.mean(np.equal(predicted_labels, labels))
