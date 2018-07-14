import tensorflow as tf


def build_graph(images, labels, loss_function, inference_function, learning_rate, global_step):
    optimizer_net = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.95)
    tf.summary.scalar('learning_rate', learning_rate)
    with tf.variable_scope(tf.get_variable_scope()) as scope:
        logits_train = inference_function(images, reuse=False, is_training=True, stohastic=True)
        probs_train = tf.nn.softmax(logits_train)
        train_loss = loss_function(logits_train, labels)

        tf.get_variable_scope().reuse_variables()

        logits_test_det = inference_function(images, reuse=True, is_training=False, stohastic=False)
        probs_test_det = tf.nn.softmax(logits_test_det)
        logits_test_stoh = inference_function(images, reuse=True, is_training=False, stohastic=True)
        probs_test_stoh = tf.nn.softmax(logits_test_stoh)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer_net.minimize(train_loss, global_step=global_step)

    return train_op, probs_train, probs_test_det, probs_test_stoh, train_loss


def build_graph_stoch(images, labels, loss_function, inference_function, learning_rate, global_step):
    optimizer_net = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.95)
    tf.summary.scalar('learning_rate', learning_rate)
    with tf.variable_scope(tf.get_variable_scope()) as scope:
        logits_stoch = inference_function(images, stochastic=True, reuse=False)
        probs_stoch = tf.nn.softmax(logits_stoch)
        tf.get_variable_scope().reuse_variables()
        logits_det = inference_function(images, stochastic=False, reuse=True)
        probs_det = tf.nn.softmax(logits_det)
        train_loss = loss_function(logits_stoch, labels)
    train_op = optimizer_net.minimize(train_loss, global_step=global_step)
    return train_op, probs_det, probs_stoch


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)
            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def build_graph_multigpu(images_train, labels_train, images_test, labels_test, global_step, loss_function,
                         accuracy_function, inference_function, learning_rate, devices):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.95)
    train_queue = tf.contrib.slim.prefetch_queue.prefetch_queue([images_train, labels_train],
                                                                capacity=20 * len(devices), num_threads=len(devices))
    test_queue = tf.contrib.slim.prefetch_queue.prefetch_queue([images_test, labels_test],
                                                               capacity=20 * len(devices), num_threads=len(devices))
    tower_grads = []
    train_loss_arr = []
    train_acc_arr = []
    test_loss_arr = []
    test_acc_arr = []
    with tf.variable_scope(tf.get_variable_scope()) as scope:
        for dev_id in range(len(devices)):
            with tf.device(devices[dev_id]):
                with tf.name_scope('tower_%s' % devices[dev_id][-1]) as scope:
                    # train ops
                    batch_images_train, batch_labels_train = train_queue.dequeue()
                    train_preds = inference_function(batch_images_train, reuse=dev_id != 0, is_training=True)
                    train_loss = loss_function(train_preds, batch_labels_train, reuse=dev_id != 0)
                    train_loss_arr.append(train_loss)
                    train_acc_arr.append(accuracy_function(train_preds, batch_labels_train))
                    variables = filter(lambda v: 'optimizer' not in v.name.lower(), tf.trainable_variables())
                    grads = optimizer.compute_gradients(train_loss, variables)
                    tower_grads.append(grads)
                    tf.get_variable_scope().reuse_variables()

                    # test ops
                    batch_images_test, batch_labels_test = test_queue.dequeue()
                    test_preds = inference_function(batch_images_test, reuse=True, is_training=False)
                    test_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=test_preds,
                                                                               labels=batch_labels_test)
                    test_loss = tf.reduce_mean(test_loss)
                    test_loss_arr.append(test_loss)
                    test_acc_arr.append(accuracy_function(test_preds, batch_labels_test))
                    tf.get_variable_scope().reuse_variables()

    grads = average_gradients(tower_grads)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_loss_op = tf.add_n(train_loss_arr)/len(devices)
    tf.summary.scalar('train loss', train_loss_op)
    train_acc_op = tf.add_n(train_acc_arr)/len(devices)
    tf.summary.scalar('train accuracy', train_acc_op)
    test_loss_op = tf.add_n(test_loss_arr)/len(devices)
    test_acc_op = tf.add_n(test_acc_arr)/len(devices)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.apply_gradients(grads, global_step=global_step)
    return train_op, test_acc_op, test_loss_op

def build_graph_multigpu_stoch(images_train, labels_train, images_test, labels_test, global_step, loss_function,
                         accuracy_function, inference_function, learning_rate, devices):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.95)
    train_queue = tf.contrib.slim.prefetch_queue.prefetch_queue([images_train, labels_train],
                                                                capacity=20 * len(devices), num_threads=len(devices))
    test_queue = tf.contrib.slim.prefetch_queue.prefetch_queue([images_test, labels_test],
                                                               capacity=20 * len(devices), num_threads=len(devices))
    tower_grads = []
    train_loss_arr = []
    train_acc_arr = []
    test_loss_arr = []
    test_acc_arr = []
    with tf.variable_scope(tf.get_variable_scope()) as scope:
        for dev_id in range(len(devices)):
            with tf.device(devices[dev_id]):
                with tf.name_scope('tower_%s' % devices[dev_id][-1]) as scope:
                    # train ops
                    batch_images_train, batch_labels_train = train_queue.dequeue()
                    train_preds = inference_function(batch_images_train, reuse=dev_id != 0, is_training=True)
                    train_loss = loss_function(train_preds, batch_labels_train, reuse=dev_id != 0)
                    train_loss_arr.append(train_loss)
                    train_acc_arr.append(accuracy_function(train_preds, batch_labels_train))
                    variables = filter(lambda v: 'optimizer' not in v.name.lower(), tf.trainable_variables())
                    grads = optimizer.compute_gradients(train_loss, variables)
                    tower_grads.append(grads)
                    tf.get_variable_scope().reuse_variables()

                    # test ops
                    batch_images_test, batch_labels_test = test_queue.dequeue()
                    test_preds = inference_function(batch_images_test, reuse=True, is_training=False)
                    test_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=test_preds,
                                                                               labels=batch_labels_test)
                    test_loss = tf.reduce_mean(test_loss)
                    test_loss_arr.append(test_loss)
                    test_acc_arr.append(accuracy_function(test_preds, batch_labels_test))
                    tf.get_variable_scope().reuse_variables()

    grads = average_gradients(tower_grads)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_loss_op = tf.add_n(train_loss_arr)/len(devices)
    tf.summary.scalar('train loss', train_loss_op)
    train_acc_op = tf.add_n(train_acc_arr)/len(devices)
    tf.summary.scalar('train accuracy', train_acc_op)
    test_loss_op = tf.add_n(test_loss_arr)/len(devices)
    test_acc_op = tf.add_n(test_acc_arr)/len(devices)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.apply_gradients(grads, global_step=global_step)
    return train_op, test_acc_op, test_loss_op


def get_weights():
    variables = tf.get_collection('variables')
    # variables = filter(lambda v: 'conv_2' in v.name.lower() or 'dense_1' in v.name.lower(), variables)
    variables = filter(lambda v: 'dense_1' in v.name.lower(), variables)
    variables = filter(lambda v: 'W' in v.name or 'kernel' in v.name, variables)
    return variables


def build_graph_with_hess(images, labels, loss_function, inference_function, learning_rate, global_step):
    optimizer_net = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.95)
    tf.summary.scalar('learning_rate', learning_rate)
    with tf.variable_scope(tf.get_variable_scope()) as scope:
        logits_stoch = inference_function(images, stochastic=True, reuse=False)
        probs_stoch = tf.nn.softmax(logits_stoch)
        tf.get_variable_scope().reuse_variables()
        logits_det = inference_function(images, stochastic=False, reuse=True)
        probs_det = tf.nn.softmax(logits_det)
        train_loss = loss_function(logits_stoch, labels)
        # weights = get_weights()
        # for v in weights:
        #     hess = tf.diag_part(tf.squeeze(tf.hessians(logits_det, v)))
        #     tf.summary.histogram(v.name + 'hessian', hess)
        #     print v.name, v.get_shape(), hess.get_shape()
    train_op = optimizer_net.minimize(train_loss, global_step=global_step)
    return train_op, probs_det, probs_stoch
