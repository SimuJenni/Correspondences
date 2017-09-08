import tensorflow as tf
import tensorflow.contrib.slim as slim
from models.AlexNet_chan_sort_2_bn1 import AlexNet

DEFAULT_FILTER_DIMS = [64, 128, 256, 512, 1024]


def cnet_argscope(activation=tf.nn.relu, kernel_size=(3, 3), padding='SAME', training=True, center=True,
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
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.convolution2d_transpose],
                        activation_fn=activation,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params,
                        weights_regularizer=slim.l2_regularizer(w_reg),
                        biases_regularizer=slim.l2_regularizer(w_reg),
                        weights_initializer=he):
        with slim.arg_scope([slim.conv2d, slim.convolution2d_transpose],
                            kernel_size=kernel_size,
                            padding=padding):
            with slim.arg_scope([slim.batch_norm], **batch_norm_params):
                with slim.arg_scope([slim.dropout], is_training=training) as arg_sc:
                    return arg_sc


class CNet:
    def __init__(self, batch_size, target_shape, pool5=True, tag='default', fix_bn=False,
                 disc_pad='VALID'):
        self.name = 'CNET_rand_{}'.format(tag)
        self.batch_size = batch_size
        self.im_shape = target_shape
        self.encoder = AlexNet(batch_size, fix_bn=fix_bn, pool5=pool5, pad=disc_pad)
        self.context_tiles = self.fake_context_tiles = self.order_pred = None
        self.pos_preds = [None]*9

    def net(self, tiles, reuse=None, train=True):

        print('tiles: {}'.format(tiles.get_shape().as_list()))
        tiles = tf.unstack(tiles, axis=1)
        tiles = tf.concat(tiles, 0)
        enc_tiles, _ = self.encoder.encode(tiles, reuse=reuse, training=train)

        tiles_encoded = tf.split(enc_tiles, 9)
        print('tiles_encoded: {}'.format(tiles_encoded[0].get_shape().as_list()))

        context = tf.stack(tiles_encoded)
        _, rand_idx = tf.nn.top_k(tf.random_uniform((9,)), k=9)
        fake_context = tf.gather(context, rand_idx)
        print('context: {}'.format(context.get_shape().as_list()))
        print('fake_context: {}'.format(fake_context.get_shape().as_list()))

        self.context_tiles = tf.unstack(context)
        self.fake_context_tiles = tf.unstack(fake_context)

        self.order_pred = self.order_classifier(reuse=reuse, training=train)

        print(self.pos_preds)
        for i, tile in enumerate(self.context_tiles):
            self.pos_preds[i] = self.position_classifier(tile, reuse=None if i == 0 else True, training=train)

        return self.order_pred

    def arrange_tiles(self, tiles):
        stacked_tiles = tf.stack(tiles, axis=1)
        stacked_shape = stacked_tiles.get_shape().as_list()
        net = tf.reshape(stacked_tiles, [stacked_shape[0], stacked_shape[1]*3, stacked_shape[2]*3, stacked_shape[3]])
        return net

    def order_classifier(self, reuse=None, training=True):
        with tf.variable_scope('order_classifier', reuse=reuse):
            with slim.arg_scope(cnet_argscope(padding='VALID', training=training)):
                real_context = tf.concat(self.context_tiles, axis=3)
                fake_context = tf.concat(self.fake_context_tiles, axis=3)
                contexts = tf.concat([real_context, fake_context], axis=0)

                net = slim.conv2d(contexts, num_outputs=2048, stride=1, kernel_size=[1, 1], scope='fc_1')
                net = slim.dropout(net, 0.5, is_training=training)
                net = slim.conv2d(net, num_outputs=2048, stride=1, kernel_size=[1, 1], scope='fc_2')
                net = slim.dropout(net, 0.5, is_training=training)
                net = slim.conv2d(net, num_outputs=2, stride=1, kernel_size=[1, 1], scope='fc_3',
                                  normalizer_fn=None, activation_fn=None)
                return slim.flatten(net)

    def position_classifier(self, tile, reuse=None, training=True):
        with tf.variable_scope('position_classifier', reuse=reuse):
            with slim.arg_scope(cnet_argscope(padding='VALID', training=training)):
                net = slim.conv2d(tile, num_outputs=256, stride=1, kernel_size=[1, 1], scope='fc_1')
                net = slim.dropout(net, 0.5, is_training=training)
                net = slim.conv2d(net, num_outputs=256, stride=1, kernel_size=[1, 1], scope='fc_2')
                net = slim.conv2d(net, num_outputs=9, stride=1, kernel_size=[1, 1], scope='fc_3',
                                  normalizer_fn=None, activation_fn=None)
                return slim.flatten(net)

    def labels_real(self):
        labels = tf.concat([tf.ones((self.batch_size,), dtype=tf.int64), tf.zeros((self.batch_size,), dtype=tf.int64)], 0)
        return tf.one_hot(labels, 2)

    def classification_loss(self):
        # Define loss for discriminator training
        loss_scope = 'classification_loss'
        classification_loss = tf.losses.softmax_cross_entropy(self.labels_real(), self.order_pred,
                                                              scope=loss_scope)
        for i, pred in enumerate(self.pos_preds):
            tf.losses.softmax_cross_entropy(tf.one_hot([i]*self.batch_size, 9), pred, scope=loss_scope,
                                            label_smoothing=1.0, weights=0.1)

        tf.summary.scalar('losses/classifier', classification_loss)
        losses = tf.losses.get_losses(loss_scope)
        losses += tf.losses.get_regularization_losses(loss_scope)
        total_loss = tf.add_n(losses, name='total_loss')

        real_pred = tf.arg_max(self.order_pred, 1)
        real_true = tf.arg_max(self.labels_real(), 1)
        tf.summary.scalar('accuracy/classifer', slim.metrics.accuracy(real_pred, real_true))
        return total_loss

    def position_loss(self):
        # Define loss for discriminator training
        loss_scope = 'position_loss'

        for i, pred in enumerate(self.pos_preds):
            pos_loss = tf.losses.softmax_cross_entropy(tf.one_hot([i]*self.batch_size, 9), pred, scope=loss_scope, weights=0.1)
            tf.summary.scalar('losses/position_{}'.format(i), pos_loss)
            real_pred = tf.arg_max(pred, 1)
            tf.summary.scalar('accuracy/position_{}'.format(i), slim.metrics.accuracy(real_pred, i*tf.ones((self.batch_size,), dtype=tf.int64)))
        losses = tf.losses.get_losses(loss_scope)
        losses += tf.losses.get_regularization_losses(loss_scope)
        total_loss = tf.add_n(losses, name='total_loss')

        return total_loss
