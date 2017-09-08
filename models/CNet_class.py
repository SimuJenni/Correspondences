import tensorflow as tf
import tensorflow.contrib.slim as slim
from models.AlexNet import AlexNet

DEFAULT_FILTER_DIMS = [64, 128, 256, 512, 1024]


def cnet_argscope(activation=tf.nn.relu, kernel_size=(3, 3), padding='SAME', training=True, center=True,
                  w_reg=0.00005, fix_bn=False):
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
    def __init__(self, batch_size, target_shape, pool5=False, tag='default', fix_bn=False,
                 disc_pad='VALID'):
        self.name = 'CNET_class_{}'.format(tag)
        self.batch_size = batch_size
        self.im_shape = target_shape
        self.encoder = AlexNet(fix_bn=fix_bn, pool5=pool5, pad=disc_pad)
        self.center_tile = self.fake_center_tile = self.context_tiles = self.fake_context_tiles = self.prediction = None

    def net(self, tiles, reuse=None, train=True):
        # print('Tiles input: {}'.format(tiles.get_shape().as_list()))
        # tiles_ = tf.unstack(tiles, axis=1)
        # tiles_ = tf.concat(tiles_, 0)
        #
        # tiles_ = tf.split(tiles_, 9)
        # center_tile_ = tiles_[4]
        # tiles_.remove(center_tile_)
        #
        # context_ = tf.stack(tiles_)
        # fake_context_ = tf.get_local_variable('fake_context_', shape=context_.get_shape())
        # fake_context_ = tf.assign(fake_context_, context_)
        # random_tile_idx = tf.random_uniform(shape=(1,), maxval=7, dtype=tf.int64)
        # fake_center_tile_ = tf.squeeze(tf.gather(fake_context_, random_tile_idx), axis=[0])
        # fake_context_ = tf.scatter_mul(fake_context_, random_tile_idx, tf.expand_dims(tf.zeros_like(center_tile_), 0))
        # fake_context_ = tf.scatter_add(fake_context_, random_tile_idx, tf.expand_dims(center_tile_, 0))
        #
        # for i, tile in enumerate(tf.unstack(fake_context_)):
        #     tf.summary.image('fake_tile/{}'.format(i), tile, max_outputs=1)
        # tf.summary.image('fake_tile/center', fake_center_tile_, max_outputs=1)
        # for i, tile in enumerate(tf.unstack(context_)):
        #     tf.summary.image('true_tile/{}'.format(i), tile, max_outputs=1)
        # tf.summary.image('true_tile/center', center_tile_, max_outputs=1)

        tiles = tf.unstack(tiles, axis=1)
        tiles = tf.concat(tiles, 0)
        enc_tiles, _ = self.encoder.encode(tiles, reuse=reuse, training=train)

        tiles_encoded = tf.split(enc_tiles, 9)
        self.center_tile = tiles_encoded[4]
        tiles_encoded.remove(self.center_tile)

        context = tf.stack(tiles_encoded)
        fake_context = tf.get_local_variable('fake_context', shape=context.get_shape())
        fake_context = tf.assign(fake_context, context)
        random_tile_idx = tf.random_uniform(shape=(1,), maxval=7, dtype=tf.int64)
        self.fake_center_tile = tf.squeeze(tf.gather(fake_context, random_tile_idx), axis=[0])
        fake_context = tf.scatter_mul(fake_context, random_tile_idx, tf.expand_dims(tf.zeros_like(self.center_tile), 0))
        fake_context = tf.scatter_add(fake_context, random_tile_idx, tf.expand_dims(self.center_tile, 0))

        self.context_tiles = tf.unstack(context)
        self.fake_context_tiles = tf.unstack(fake_context)

        self.prediction = self.order_classifier(reuse=reuse, training=train)
        return self.prediction

    def predict_center(self, reuse=None, training=True):
        with tf.variable_scope('predictor', reuse=reuse):
            with slim.arg_scope(cnet_argscope(padding='SAME', training=training)):
                context = tf.concat(self.context_tiles, axis=3)
                net = slim.conv2d(context, num_outputs=1024, stride=1, kernel_size=[3, 3], scope='conv_1')
                net = slim.conv2d(net, num_outputs=256, stride=1, kernel_size=[3, 3], scope='conv_2',
                                  normalizer_fn=None)
                return net

    def order_classifier(self, reuse=None, training=True):
        with tf.variable_scope('classifier', reuse=reuse):
            with slim.arg_scope(cnet_argscope(padding='VALID', training=training)):
                real_context = tf.concat(self.context_tiles + [self.center_tile], axis=3)
                fake_context = tf.concat(self.fake_context_tiles + [self.fake_center_tile], axis=3)
                contexts = tf.concat([real_context, fake_context], axis=0)

                net = slim.conv2d(contexts, num_outputs=4096, stride=1, kernel_size=[1, 1], scope='fc_2')
                net = slim.conv2d(net, num_outputs=2, stride=1, kernel_size=[1, 1], scope='fc_3',
                                  normalizer_fn=None, activation_fn=None)
                return slim.flatten(net)

    def labels_real(self):
        labels = tf.concat([tf.ones((self.batch_size,), dtype=tf.int64), tf.zeros((self.batch_size,), dtype=tf.int64)], 0)
        return tf.one_hot(labels, 2)

    def labels_fake(self):
        labels = tf.concat([tf.zeros((self.batch_size,), dtype=tf.int64), tf.ones((self.batch_size,), dtype=tf.int64)], 0)
        return tf.one_hot(labels, 2)

    def mpool_norm_tile(self, tile):
        tile_shape = tile.get_shape().as_list()
        tile = slim.avg_pool2d(tile, kernel_size=tile_shape[1:3], stride=1)
        tile = tf.nn.l2_normalize(tile, dim=3)
        return tile

    def classification_loss(self):
        # Define loss for discriminator training
        loss_scope = 'classification_loss'
        classification_loss = tf.losses.softmax_cross_entropy(self.labels_real(), self.prediction,
                                                    scope=loss_scope)
        tf.summary.scalar('losses/classifier', classification_loss)
        losses = tf.losses.get_losses(loss_scope)
        losses += tf.losses.get_regularization_losses(loss_scope)
        total_loss = tf.add_n(losses, name='total_loss')

        real_pred = tf.arg_max(self.prediction, 1)
        real_true = tf.arg_max(self.labels_real(), 1)
        tf.summary.scalar('accuracy/classifer', slim.metrics.accuracy(real_pred, real_true))
        return total_loss
