import tensorflow as tf


def build_net_1(input_placeholder,
                output_size,
                scope,
                size,
                scope_scope='',
                name_name='',
                activation_fn=tf.nn.relu,
                output_activation=None,
                use_bias=True,
                input_batch_norm=False,
                last_layer_init=True
                ):
    """
    Some handy function to build your nets and retrieve their weights with.
    """
    w = {}
    with tf.variable_scope(scope):
        SCOPE = scope_scope + scope
        x = input_placeholder
        if input_batch_norm:
            name = name_name + 'state_input'
            x = tf.contrib.layers.batch_norm(x, decay=0.9, center=True, scale=True,
                                             scope='bn_' + name, is_training=False)
            w[name + '_gamma'] = tf.get_default_graph().get_tensor_by_name(SCOPE + '/bn_' + name + '/gamma:0')
            w[name + '_beta'] = tf.get_default_graph().get_tensor_by_name(SCOPE + '/bn_' + name + '/beta:0')

        n = 0
        for (n_units, use_bn) in size:
            name = name_name + 'layer_%d' % n
            n += 1
            x = tf.layers.dense(inputs=x, units=n_units, activation=None, use_bias=use_bias, name=name)
            w[name + '_kernel'] = tf.get_default_graph().get_tensor_by_name(SCOPE + '/' + name + '/kernel:0')
            if use_bias: w[name + '_bias'] = tf.get_default_graph().get_tensor_by_name(SCOPE + '/' + name + '/bias:0')
            if use_bn:
                x = tf.contrib.layers.batch_norm(x, decay=0.9, center=True, scale=True,
                                                 scope='bn_' + name, is_training=False)
                w[name + '_gamma'] = tf.get_default_graph().get_tensor_by_name(SCOPE + '/bn_' + name + '/gamma:0')
                w[name + '_beta'] = tf.get_default_graph().get_tensor_by_name(SCOPE + '/bn_' + name + '/beta:0')
            x = activation_fn(x)

        name = name_name + 'output_layer'
        if last_layer_init:
            x = tf.layers.dense(inputs=x, units=output_size, activation=output_activation, use_bias=use_bias, name=name,
                                kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
        else:
            x = tf.layers.dense(inputs=x, units=output_size, activation=output_activation, use_bias=use_bias, name=name)
        w[name + '_kernel'] = tf.get_default_graph().get_tensor_by_name(SCOPE + '/' + name + '/kernel:0')
        if use_bias: w[name + '_bias'] = tf.get_default_graph().get_tensor_by_name(SCOPE + '/' + name + '/bias:0')

    return x, w


def build_net_2(input_placeholder_1,
                input_placeholder_2,
                output_size,
                scope,
                size,
                ac_size,
                junction=1,
                activation_fn=tf.nn.relu,
                output_activation=None,
                use_bias=True,
                input_batch_norm=False
                ):
    """
    Some other handy function to build your two-input nets and retrieve their weights with.
    """
    w = {}
    with tf.variable_scope(scope):

        x = input_placeholder_1
        if input_batch_norm:
            name = 'state_input'
            x = tf.contrib.layers.batch_norm(x, decay=0.9, center=True, scale=True,
                                             scope='bn_' + name, is_training=False)
            w[name + '_gamma'] = tf.get_default_graph().get_tensor_by_name(scope + '/bn_' + name + '/gamma:0')
            w[name + '_beta'] = tf.get_default_graph().get_tensor_by_name(scope + '/bn_' + name + '/beta:0')

        y, w_y = build_net_1(input_placeholder_2, size=ac_size, output_size=size[junction][0],
                             scope='y_net', scope_scope=scope+'/', name_name='y_net_',
                             activation_fn=activation_fn, output_activation=None,
                             use_bias=False, input_batch_norm=input_batch_norm, last_layer_init=False)

        n = 0
        for (n_units, use_bn) in size:
            name = 'layer_%d' % n
            x = tf.layers.dense(inputs=x, units=n_units, activation=None, use_bias=use_bias, name=name)
            w[name + '_kernel'] = tf.get_default_graph().get_tensor_by_name(scope + '/' + name + '/kernel:0')
            if use_bias: w[name + '_bias'] = tf.get_default_graph().get_tensor_by_name(scope + '/' + name + '/bias:0')
            if use_bn:
                x = tf.contrib.layers.batch_norm(x, decay=0.9, center=True, scale=True,
                                                 scope='bn_' + name, is_training=False)
                w[name + '_gamma'] = tf.get_default_graph().get_tensor_by_name(scope + '/bn_' + name + '/gamma:0')
                w[name + '_beta'] = tf.get_default_graph().get_tensor_by_name(scope + '/bn_' + name + '/beta:0')
            if n == junction:
                x += y
            x = activation_fn(x)
            n += 1

        name = 'output_layer'
        x = tf.layers.dense(inputs=x, units=output_size, activation=output_activation, use_bias=use_bias, name=name,
                            kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
        w[name + '_kernel'] = tf.get_default_graph().get_tensor_by_name(scope + '/' + name + '/kernel:0')
        if use_bias: w[name + '_bias'] = tf.get_default_graph().get_tensor_by_name(scope + '/' + name + '/bias:0')

    # union over weights
    for key in w_y.keys():
        w[key] = w_y[key]

    return x, w
