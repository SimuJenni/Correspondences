import tensorflow as tf
import tensorflow.contrib.slim as slim
from layers import conv_group, lrelu


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
    def __init__(self, fc_activation=lrelu, fix_bn=False, pool5=True, pad='VALID'):
        self.fix_bn = fix_bn
        self.fc_activation = fc_activation
        self.use_pool5 = pool5
        self.pad = pad

    def encode(self, net, reuse=None, training=True):
        """Builds a discriminator network on top of inputs.

        Args:
            net: Input to the discriminator
            reuse: Whether to reuse already defined variables
            training: Whether in train or eval mode.
            with_fc: Whether to include fully connected layers (used during unsupervised training)

        Returns:
            Resulting logits
        """
        with tf.variable_scope('encoder', reuse=reuse):
            with slim.arg_scope(alexnet_argscope(activation=self.fc_activation, padding='SAME', training=training,
                                                 fix_bn=self.fix_bn)):
                net = slim.conv2d(net, 96, kernel_size=[11, 11], stride=4, scope='conv_1', padding=self.pad,
                                  normalizer_fn=None)
                net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope='pool_1', padding=self.pad)
                net = tf.nn.lrn(net, depth_radius=2, alpha=0.00002, beta=0.75)
                net = conv_group(net, 256, kernel_size=[5, 5], scope='conv_2')
                net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope='pool_2', padding=self.pad)
                net = tf.nn.lrn(net, depth_radius=2, alpha=0.00002, beta=0.75)
                net = slim.conv2d(net, 384, kernel_size=[3, 3], scope='conv_3')
                net = conv_group(net, 384, kernel_size=[3, 3], scope='conv_4')
                net = conv_group(net, 256, kernel_size=[3, 3], scope='conv_5')
                if self.use_pool5:
                    net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope='pool_5', padding=self.pad)
                encoded = net
                net_shape = net.get_shape().as_list()
                net = slim.conv2d(net, num_outputs=512, stride=1, kernel_size=net_shape[1:3], scope='fc_1',
                                  padding='VALID')
                return net, encoded
