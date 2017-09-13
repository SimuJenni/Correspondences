import tensorflow as tf
import tensorflow.contrib.slim as slim
from models.AlexNet_chan_sort_16 import AlexNet

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
        self.name = 'CNET_center_{}'.format(tag)
        self.batch_size = batch_size
        self.im_shape = target_shape
        self.encoder = AlexNet(batch_size, fix_bn=fix_bn, pool5=pool5, pad=disc_pad)
        self.context_tiles = self.fake_context = self.order_pred = None
        self.pos_preds = [None]*9
        self.num_neg = 4

    def put_together(self, tiles):
        in_shape = tiles[0].get_shape().as_list()
        out_shape = [in_shape[0], in_shape[1]*3, in_shape[2]*3, in_shape[3]]
        idx = tf.meshgrid(tf.range(in_shape[0]), tf.range(in_shape[1]), tf.range(in_shape[2]),
                          tf.range(in_shape[3]), indexing='ij')
        idx = tf.transpose(idx, perm=[1, 2, 3, 4, 0])

        tile_res = []
        for y in range(3):
            for x in range(3):
                i = 3*y + x
                tile_res.append(tf.scatter_nd(idx+[0, y*in_shape[1], x*in_shape[2], 0], tiles[i], shape=out_shape))

        res = sum(tile_res)
        tf.summary.image('input/img', res, max_outputs=3)

    def match(self, tiles, img, reuse=None, train=True):
        tiles = tf.unstack(tiles, axis=1)
        tiles = tf.concat(tiles, 0)
        enc_tiles, _ = self.encoder.encode(tiles, reuse=reuse, training=train)

        context_tiles = tf.split(enc_tiles, 9)
        context_pre = tf.concat(context_tiles[:4], 3)
        context_post = tf.concat(context_tiles[5:], 3)

        print('Context-pre: {}'.format(context_pre.get_shape().as_list()))
        print('Context-post: {}'.format(context_post.get_shape().as_list()))

        enc_img, _ = self.encoder.encode(img, reuse=True, training=train)
        enc_shape = enc_img.get_shape().as_list()
        print('enc:img: {}'.format(enc_img.get_shape().as_list()))
        context_pre = tf.tile(context_pre, [1, enc_shape[1], enc_shape[2], 1])
        context_post = tf.tile(context_post, [1, enc_shape[1], enc_shape[2], 1])

        fmap = tf.concat([context_pre, enc_img, context_post], 3)
        pred = self.center_classifier(fmap, reuse=reuse, training=train)
        return pred

    def net(self, tiles, reuse=None, train=True):

        tiles = tf.unstack(tiles, axis=1)
        self.put_together(tiles)

        tiles = tf.concat(tiles, 0)
        enc_tiles, _ = self.encoder.encode(tiles, reuse=reuse, training=train)

        c_tiles = tf.split(enc_tiles, 9)
        context = tf.concat([c_tiles[1], c_tiles[3], c_tiles[5], c_tiles[7]], 3)

        pos = tf.concat([context, c_tiles[4]], 3)
        negs = [tf.concat([context, c_tiles[0]], 3), tf.concat([context, c_tiles[2]], 3),
                tf.concat([context, c_tiles[6]], 3), tf.concat([context, c_tiles[8]], 3)]

        contexts = tf.concat([pos] + negs, axis=0)
        order_pred = self.center_classifier(contexts, reuse=reuse, training=train)
        self.order_pred = slim.flatten(order_pred)

        for i, tile in enumerate(c_tiles):
            self.pos_preds[i] = self.position_classifier(tile, reuse=None if i == 0 else True, training=train)

        return self.order_pred

    def center_classifier(self, contexts, reuse=None, training=True):
        with tf.variable_scope('order_classifier', reuse=reuse):
            with slim.arg_scope(cnet_argscope(padding='VALID', training=training)):
                net = slim.conv2d(contexts, num_outputs=1024, stride=1, kernel_size=[1, 1], scope='fc_1')
                net = slim.dropout(net)
                net = slim.conv2d(net, num_outputs=1024, stride=1, kernel_size=[1, 1], scope='fc_2')
                net = slim.dropout(net)
                net = slim.conv2d(net, num_outputs=1, stride=1, kernel_size=[1, 1], scope='fc_3',
                                  normalizer_fn=None, activation_fn=None)
                return net

    def position_classifier(self, tile, reuse=None, training=True):
        with tf.variable_scope('position_classifier', reuse=reuse):
            with slim.arg_scope(cnet_argscope(padding='VALID', training=training)):
                net = slim.conv2d(tile, num_outputs=256, stride=1, kernel_size=[1, 1], scope='fc_1')
                net = slim.conv2d(net, num_outputs=9, stride=1, kernel_size=[1, 1], scope='fc_3',
                                  normalizer_fn=None, activation_fn=None)
                return slim.flatten(net)

    def labels(self):
        labels = tf.concat([tf.ones((self.batch_size, 1), dtype=tf.int64),
                            tf.zeros((self.num_neg*self.batch_size, 1), dtype=tf.int64)], 0)
        return labels

    def weights(self):
        labels = tf.concat([tf.ones((self.batch_size, 1), dtype=tf.int64)*self.num_neg,
                            tf.ones((self.num_neg*self.batch_size, 1), dtype=tf.int64)], 0)
        return labels

    def classification_loss(self):
        # Define loss for discriminator training
        loss_scope = 'classification_loss'
        classification_loss = tf.losses.sigmoid_cross_entropy(self.labels(), self.order_pred,
                                                              scope=loss_scope, weights=self.weights())
        for i, pred in enumerate(self.pos_preds):
            tf.losses.softmax_cross_entropy(tf.one_hot([i]*self.batch_size, 9), pred, scope=loss_scope, weights=-0.5)

        tf.summary.scalar('losses/classifier', classification_loss)
        losses = tf.losses.get_losses(loss_scope)
        losses += tf.losses.get_regularization_losses(loss_scope)
        total_loss = tf.add_n(losses, name='total_loss')

        real_pred = tf.to_int64(tf.round(self.order_pred))
        real_true = self.labels()
        tf.summary.scalar('accuracy/classifer', slim.metrics.accuracy(real_pred, real_true))
        return total_loss

    def position_loss(self):
        # Define loss for discriminator training
        loss_scope = 'position_loss'
        for i, pred in enumerate(self.pos_preds):
            pos_loss = tf.losses.softmax_cross_entropy(tf.one_hot([i]*self.batch_size, 9), pred,
                                                       scope=loss_scope, weights=0.5)
            tf.summary.scalar('losses/position_{}'.format(i), pos_loss)
            real_pred = tf.arg_max(pred, 1)
            tf.summary.scalar('accuracy/position_{}'.format(i),
                              slim.metrics.accuracy(real_pred, i*tf.ones((self.batch_size,), dtype=tf.int64)))
        losses = tf.losses.get_losses(loss_scope)
        losses += tf.losses.get_regularization_losses(loss_scope)
        total_loss = tf.add_n(losses, name='total_loss')
        return total_loss
