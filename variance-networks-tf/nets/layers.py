import tensorflow as tf


def _variable_on_cpu(name, shape, initializer, dtype):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def initialize_weights_with_loss(name, shape, initializer, dtype, wd):
    var = _variable_on_cpu(name, shape, initializer, dtype)
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('l2_loss', weight_decay)
    return var


def initialize_variable(name, shape, initializer, dtype):
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


# BASE LAYERS
def dense_layer(input_tensor, num_inputs, num_outputs, nonlinearity, wd, name):
    with tf.variable_scope(name) as scope:
        W = initialize_weights_with_loss('W',
                                         [num_inputs, num_outputs],
                                         tf.truncated_normal_initializer(stddev=1e-2, seed=322), 
                                         dtype=tf.float32, 
                                         wd=wd)
        output = tf.matmul(input_tensor, W)
        biases = _variable_on_cpu('biases', [num_outputs], tf.constant_initializer(0.0), dtype=tf.float32)
        output = tf.nn.bias_add(output, biases)
        if nonlinearity:
            output = nonlinearity(output, name=scope.name)
    return output


def conv_2d_layer(input_tensor, filter_shape, input_channels, output_channels, nonlinearity, wd, padding, name,
                  with_biases=True):
    with tf.variable_scope(name) as scope:
        kernel = initialize_weights_with_loss('kernel',
                                              [filter_shape[0], filter_shape[1], input_channels, output_channels],
                                              tf.contrib.layers.xavier_initializer(seed=322),
                                              dtype=tf.float32,
                                              wd=wd)
        output = tf.nn.conv2d(input_tensor, kernel, [1, 1, 1, 1], padding=padding)
        if with_biases:
            biases = _variable_on_cpu('biases', [output_channels], tf.constant_initializer(0.0), dtype=tf.float32)
            output = tf.nn.bias_add(output, biases)
        if nonlinearity:
            output = nonlinearity(output, name=scope.name)
    return output


# FULLY VARIANCE LAYERS
def fully_variance_dense(input_tensor, num_inputs, num_outputs, mean_initializer, name, stochastic=True, reuse=False):
    with tf.variable_scope(name) as scope:
        W = tf.get_variable('W', [num_inputs, num_outputs], initializer=mean_initializer, dtype=tf.float32,
                            trainable=False)
        log_sigma2 = tf.get_variable('log_sigma2', [num_inputs, num_outputs],
                                     initializer=tf.constant_initializer(-3.0),
                                     dtype=tf.float32, trainable=True)
        mu = tf.matmul(input_tensor, W)
        si = tf.sqrt(tf.matmul(input_tensor * input_tensor, tf.exp(log_sigma2)) + 1e-16)
        output = mu
        if stochastic:
            output += tf.random_normal(mu.shape, mean=0, stddev=1) * si

        # summaries
        if not reuse:
            error = 0.5*(1.0+tf.erf((-mu)/tf.sqrt(2.0)/si))
            tf.summary.scalar('error', tf.reduce_sum(error))
            #tf.summary.histogram('log_sigma2', log_sigma2)
    return output


def fully_variance_conv_2d(input_tensor, filter_shape, input_channels, output_channels, mean_initializer, padding,
                           name, stochastic=True, reuse=False):
    with tf.variable_scope(name) as scope:
        kernel = tf.get_variable('kernel',
                                 [filter_shape[0], filter_shape[1], input_channels, output_channels],
                                 initializer=mean_initializer, dtype=tf.float32, trainable=False)
        log_sigma2 = tf.get_variable('log_sigma2', [filter_shape[0], filter_shape[1], input_channels, output_channels],
                                     initializer=tf.constant_initializer(-3.0),
                                     dtype=tf.float32, trainable=True)
        conved_mu = tf.nn.conv2d(input_tensor, kernel, [1, 1, 1, 1], padding=padding)
        conved_si = tf.sqrt(tf.nn.conv2d(input_tensor * input_tensor,
                                         tf.exp(log_sigma2), [1, 1, 1, 1],
                                         padding=padding)+1e-16)
        output = conved_mu
        if stochastic:
            output += tf.random_normal(conved_mu.shape, mean=0, stddev=1) * conved_si

        # summaries
        if not reuse:
            error = 0.5*(1.0+tf.erf((-conved_mu)/tf.sqrt(2.0)/conved_si))
            tf.summary.scalar('error', tf.reduce_sum(error))
            #tf.summary.histogram('log_sigma2', log_sigma2)
    return output


# PHASE TRANSITION LAYERS
def pt_dense(input_tensor, num_inputs, num_outputs, name, stochastic=True, with_bias=True, reuse=False):
    with tf.variable_scope(name) as scope:
        W = tf.get_variable('W', [num_inputs, num_outputs], initializer=tf.truncated_normal_initializer(1e-2),
                            dtype=tf.float32, trainable=True)
        log_alpha = tf.get_variable('log_alpha', [], initializer=tf.constant_initializer(-10.0), dtype=tf.float32,
                                    trainable=True)
        log_alpha = tf.clip_by_value(log_alpha, -20.0, 20.0)

        if not reuse:
            # computing reg
            k1, k2, k3 = 0.63576, 1.8732, 1.48695
            C = -k1
            mdkl = k1 * tf.nn.sigmoid(k2 + k3 * log_alpha) - 0.5 * tf.log1p(tf.exp(-log_alpha)) + C
            kl = -tf.reduce_sum(mdkl) * tf.reduce_prod(tf.cast(W.get_shape(), tf.float32))
            tf.add_to_collection('kl_loss', kl)

        # computing output
        mu = tf.matmul(input_tensor, W)
        si = tf.sqrt(tf.matmul(input_tensor * input_tensor, tf.exp(log_alpha) * W * W)   + 1e-16)
        output = mu
        if stochastic:
            output += tf.random_normal(mu.shape, mean=0, stddev=1) * si
        if with_bias:
            biases = tf.get_variable('biases', num_outputs, tf.float32, tf.constant_initializer(0.0))
            output = tf.nn.bias_add(output, biases)

        # summaries
        if not reuse:
            if with_bias:
                error = 0.5*(1.0+tf.erf((-mu-biases)/tf.sqrt(2.0)/si))
            else:
                error = 0.5*(1.0+tf.erf((-mu)/tf.sqrt(2.0)/si))
            tf.summary.scalar('error', tf.reduce_sum(error))
            tf.summary.scalar('log_alpha', log_alpha)
            tf.add_to_collection('log_alpha', log_alpha)
    return output


def pt_conv_2d(input_tensor, filter_shape, input_channels, output_channels, padding, name, stochastic=True,
               with_bias=True, reuse=False):
    with tf.variable_scope(name) as scope:
        kernel = tf.get_variable('kernel', [filter_shape[0], filter_shape[1], input_channels, output_channels],
                                 initializer=tf.contrib.layers.xavier_initializer(seed=322), dtype=tf.float32,
                                 trainable=True)
        log_alpha = tf.get_variable('log_alpha', [], initializer=tf.constant_initializer(-10.0), dtype=tf.float32,
                                    trainable=True)
        log_alpha = tf.clip_by_value(log_alpha, -20.0, 20.0)

        if not reuse:
            # computing reg
            k1, k2, k3 = 0.63576, 1.8732, 1.48695
            C = -k1
            mdkl = k1 * tf.nn.sigmoid(k2 + k3 * log_alpha) - 0.5 * tf.log1p(tf.exp(-log_alpha)) + C
            kl = -tf.reduce_sum(mdkl) * tf.reduce_prod(tf.cast(kernel.get_shape(), tf.float32))
            tf.add_to_collection('kl_loss', kl)

        # computing output
        conved_mu = tf.nn.conv2d(input_tensor, kernel, [1, 1, 1, 1], padding=padding)
        conved_si = tf.sqrt(tf.nn.conv2d(input_tensor * input_tensor,
                                         tf.exp(log_alpha) * kernel * kernel,
                                         [1, 1, 1, 1], padding=padding)+1e-16)
        output = conved_mu
        if stochastic:
            output += tf.random_normal(conved_mu.shape, mean=0, stddev=1) * conved_si
        if with_bias:
            biases = tf.get_variable('biases', output_channels, tf.float32, tf.constant_initializer(0.0))
            output = tf.nn.bias_add(output, biases)

        # summaries
        if not reuse:
            if with_bias:
                error = 0.5*(1.0+tf.erf((-conved_mu-biases)/tf.sqrt(2.0)/conved_si))
            else:
                error = 0.5*(1.0+tf.erf((-conved_mu)/tf.sqrt(2.0)/conved_si))
            tf.summary.scalar('error', tf.reduce_sum(error))
            tf.summary.scalar('log_alpha', log_alpha)
            tf.add_to_collection('log_alpha', log_alpha)

    return output
