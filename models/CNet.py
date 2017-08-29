import tensorflow as tf
import tensorflow.contrib.slim as slim
from models.AlexNet import AlexNet

DEFAULT_FILTER_DIMS = [64, 128, 256, 512, 1024]


def cnet_argscope(activation=tf.nn.elu, kernel_size=(3, 3), padding='SAME', training=True, center=True,
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
    def __init__(self, batch_size, target_shape, pool5=False, tag='default', fix_bn=False,
                 disc_pad='VALID'):
        self.name = 'CNET_{}'.format(tag)
        self.batch_size = batch_size
        self.im_shape = target_shape
        self.encoder = AlexNet(fix_bn=fix_bn, pool5=pool5, pad=disc_pad)
        self.center_tile = self.context_tiles = self.center_prediction = None

    def net(self, tiles, reuse=None, train=True):
        tiles = tf.unstack(tiles, axis=1)
        print(tiles[0].get_shape().as_list())
        tiles_encoded = []
        for i, tile in enumerate(tiles):
            enc_tile = self.encoder.encode(tile, reuse=reuse if i == 0 else True, training=train)
            tiles_encoded.append(enc_tile)

        self.center_tile = tiles_encoded[4]
        tiles_encoded.remove(self.center_tile)
        self.context_tiles = tiles_encoded
        self.center_prediction = self.predict_center(reuse=reuse, training=train)
        return self.center_prediction

    def predict_center(self, reuse=None, training=True):
        with tf.variable_scope('predictor', reuse=reuse):
            with slim.arg_scope(cnet_argscope(padding='SAME', training=training)):
                context = tf.concat(self.context_tiles, axis=3)
                net = slim.conv2d(context, num_outputs=1024, stride=1, kernel_size=[3, 3], scope='conv_1')
                net = slim.conv2d(net, num_outputs=256, stride=1, kernel_size=[3, 3], scope='conv_2',
                                  normalizer_fn=None)
                return net

    def order_classifier(self, tiles, reuse=None, training=True):
        with tf.variable_scope('predictor', reuse=reuse):
            with slim.arg_scope(cnet_argscope(padding='SAME', training=training)):
                net = slim.conv2d(tiles, num_outputs=4096, stride=1, kernel_size=[3, 3], scope='fc_1', padding='VALID')
                net = slim.dropout(net, 0.5, is_training=training)
                net = slim.conv2d(net, num_outputs=2, stride=1, kernel_size=[1, 1], scope='fc_2',
                                  normalizer_fn=None)
                return net

    def mpool_norm_tile(self, tile):
        tile_shape = tile.get_shape().as_list()
        tile = slim.avg_pool2d(tile, kernel_size=tile_shape[1:3], stride=1)
        tile = tf.nn.l2_normalize(tile, dim=3)
        return tile

    def prediction_loss(self):
        scope = 'prediction_loss'
        pred = self.mpool_norm_tile(self.center_prediction)
        center = self.mpool_norm_tile(self.center_tile)
        pred_loss = tf.losses.cosine_distance(predictions=pred, labels=center, dim=3, scope=scope)
        tf.summary.scalar('losses/prediction', pred_loss)
        losses = tf.losses.get_losses(scope)
        losses += tf.losses.get_regularization_losses(scope)
        pred_loss = tf.add_n(losses, name='pred_total_loss')
        return pred_loss

    def contrastive_loss(self):
        scope = 'contrastive_loss'
        contrast_loss = 0
        center = self.mpool_norm_tile(self.center_tile)
        num_context = len(self.context_tiles)
        for tile in self.context_tiles:
            tile = self.mpool_norm_tile(tile)
            contrast_loss += tf.losses.cosine_distance(predictions=center,
                                                        labels=tile,
                                                        dim=3,
                                                        weights=-1.0 / num_context,
                                                        scope=scope)
        tf.summary.scalar('losses/contrastive', contrast_loss)
        losses = tf.losses.get_losses(scope)
        losses += tf.losses.get_regularization_losses(scope)
        contrast_loss = tf.add_n(losses, name='contrast_total_loss')
        return contrast_loss
