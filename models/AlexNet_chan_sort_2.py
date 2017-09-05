import tensorflow as tf
import tensorflow.contrib.slim as slim
from layers import conv_group


def alexnet_argscope(activation=tf.nn.relu, kernel_size=(3, 3), padding='SAME', training=True, center=True,
                      w_reg=0.0001, fix_bn=False):
    """Defines default parameter values for all the layers used in ToonNet.

    Args:
        activation: The default activation function
        kernel_size: The default kernel size for convolution layers
        padding: The default border mode
        training: Whether in train or eval mode
        center: Whether to use centering in batchnorm
        w_reg: Parameter for weight-decay

    Returns:
        An argscope
    """
    train_bn = training and not fix_bn
    batch_norm_params = {
        'is_training': train_bn,
        'decay': 0.99,
        'epsilon': 0.001,
        'center': center,
        'fused': True
    }
    he = tf.contrib.layers.variance_scaling_initializer(mode='FAN_AVG')
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=activation,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params,
                        weights_regularizer=slim.l2_regularizer(w_reg),
                        biases_regularizer=slim.l2_regularizer(w_reg),
                        weights_initializer=he):
        with slim.arg_scope([slim.conv2d],
                            kernel_size=kernel_size,
                            padding=padding):
            with slim.arg_scope([slim.batch_norm], **batch_norm_params):
                with slim.arg_scope([slim.dropout], is_training=training) as arg_sc:
                    return arg_sc


class AlexNet:
    def __init__(self, batch_size, fc_activation=tf.nn.relu, fix_bn=False, pool5=True, pad='VALID'):
        self.fix_bn = fix_bn
        self.fc_activation = fc_activation
        self.use_pool5 = pool5
        self.pad = pad
        self.batch_size = batch_size
        self.num_layers = 5
        self.name = 'AlexNet_sorted2'

    def classifier(self, net, num_classes, reuse=None, training=True):
        """Builds a discriminator network on top of inputs.

        Args:
            net: Input to the discriminator
            reuse: Whether to reuse already defined variables
            training: Whether in train or eval mode.
            with_fc: Whether to include fully connected layers (used during unsupervised training)

        Returns:
            Resulting logits
        """
        layers = []
        with tf.variable_scope('discriminator', reuse=reuse):
            with slim.arg_scope(alexnet_argscope(activation=self.fc_activation, padding='SAME', training=training,
                                                 fix_bn=self.fix_bn)):
                net = slim.conv2d(net, 96, kernel_size=[11, 11], stride=4, scope='conv_1', padding=self.pad,
                                  normalizer_fn=None)
                net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope='pool_1', padding=self.pad)
                net = group_sort(net, 8)
                layers.append(net)
                net = tf.nn.lrn(net, depth_radius=2, alpha=0.00002, beta=0.75)
                net = conv_group(net, 256, kernel_size=[5, 5], scope='conv_2')
                net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope='pool_2', padding=self.pad)
                layers.append(net)
                net = tf.nn.lrn(net, depth_radius=2, alpha=0.00002, beta=0.75)
                net = slim.conv2d(net, 384, kernel_size=[3, 3], scope='conv_3')
                layers.append(net)
                net = conv_group(net, 384, kernel_size=[3, 3], scope='conv_4')
                layers.append(net)
                net = conv_group(net, 256, kernel_size=[3, 3], scope='conv_5')
                net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope='pool_5', padding=self.pad)
                layers.append(net)

        with tf.variable_scope('fully_connected', reuse=reuse):
            with slim.arg_scope(
                    alexnet_argscope(activation=self.fc_activation, padding='SAME', training=training,
                                     fix_bn=self.fix_bn)):
                net = slim.flatten(net)
                net = slim.fully_connected(net, 4096, scope='fc1')
                net = slim.dropout(net, 0.5, is_training=training)
                net = slim.fully_connected(net, 4096, scope='fc2')
                net = slim.dropout(net, 0.5, is_training=training)
                net = slim.fully_connected(net, num_classes, scope='fc3',
                                           activation_fn=None,
                                           normalizer_fn=None,
                                           biases_initializer=tf.zeros_initializer())

                return net, layers


def group_sort(net, num_groups=4):
    groups = tf.split(net, num_groups, axis=3)
    groups_sorted = []
    for g in groups:
        groups_sorted.append(sort_channels_4(g))
    return tf.concat(groups_sorted, axis=3)


def sort_channels_4(net):
    net_shape = net.get_shape().as_list()
    print('net: {}'.format(net.get_shape().as_list()))
    fmap_energy = slim.avg_pool2d(net, net_shape[1:3])
    print('energy: {}'.format(fmap_energy.get_shape().as_list()))
    _, ch_idx = tf.nn.top_k(fmap_energy, k=net_shape[3])
    ch_idx = tf.tile(ch_idx, [1, net_shape[1], net_shape[2], 1])
    print('ch_idx: {}'.format(ch_idx.get_shape().as_list()))

    aux_idx = tf.meshgrid(tf.range(net_shape[0]), tf.range(net_shape[1]), tf.range(net_shape[2]), tf.range(net_shape[3]), indexing='ij')
    print('aux_idx: {}'.format(aux_idx[0].get_shape().as_list()))

    idx = tf.stack(aux_idx[:-1]+[ch_idx], axis=-1)
    print('idx: {}'.format(idx.get_shape().as_list()))

    net_sorted = tf.gather_nd(net, idx)
    print('net_sorted: {}'.format(net_sorted.get_shape().as_list()))

    return net_sorted