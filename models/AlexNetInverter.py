import tensorflow as tf
import tensorflow.contrib.slim as slim
from AlexNet_layers import AlexNet
from layers import up_conv2d, lrelu
from utils import montage_tf

DEFAULT_FILTER_DIMS = [64, 128, 256, 384, 384]


def toon_net_argscope(activation=lrelu, kernel_size=(3, 3), padding='SAME', training=True, center=True,
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


class AlexNetInverter:
    def __init__(self, model, target_shape, layer_id=1, tag='default'):
        """Initialises a ToonNet using the provided parameters.

        Args:
            num_layers: The number of convolutional down/upsampling layers to be used.
            batch_size: The batch-size used during training (used to generate training labels)
        """
        self.name = 'AlexNetInverter_{}_layer{}'.format(tag, layer_id)
        self.num_layers = 5
        self.batch_size = model.batch_size
        self.im_shape = target_shape
        self.layer_id = layer_id
        self.encoder = model
        self.dec_im = self.disc_out = self.enc_im = self.enc_dec_im = None

    def net(self, imgs, reuse=None, train=True):
        """Builds the full ToonNet architecture with the given inputs.

        Args:
            imgs: Placeholder for input images
            reuse: Whether to reuse already defined variables.
            train: Whether in train or eval mode

        Returns:
            dec_im: The autoencoded image
            dec_gen: The reconstructed image from cartoon and edge inputs
            disc_out: The discriminator output
            enc_im: Encoding of the image
            gen_enc: Output of the generator
        """
        # Concatenate cartoon and edge for input to generator
        _, enc_layers = self.encoder.classifier(imgs, 1000, reuse=reuse, training=train)
        self.enc_im = enc_layers[self.layer_id]

        # Decode both encoded images and generator output using the same decoder
        self.dec_im = self.decoder(self.enc_im, self.layer_id, reuse=reuse, training=train)
        _, dec_enc_layers = self.encoder.classifier(self.dec_im, 1000, reuse=True, training=train)
        self.enc_dec_im = dec_enc_layers[self.layer_id]

        # Build input for discriminator (discriminator tries to guess order of real/fake)
        disc_in = tf.concat([imgs, self.dec_im], 0)
        self.disc_out = self.discriminator(disc_in, reuse=reuse, training=train)

        return self.dec_im

    def labels_real(self):
        labels = tf.concat([tf.ones((self.batch_size,), dtype=tf.int64), tf.zeros((self.batch_size,), dtype=tf.int64)], 0)
        return tf.one_hot(labels, 2)

    def labels_fake(self):
        labels = tf.concat([tf.zeros((self.batch_size,), dtype=tf.int64), tf.ones((self.batch_size,), dtype=tf.int64)], 0)
        return tf.one_hot(labels, 2)

    def classifier(self, img, num_classes, reuse=None, training=True):
        """Builds a classifier on top either the encoder, generator or discriminator trained in the AEGAN.

        Args:
            img: Input image
            num_classes: Number of output classes
            reuse: Whether to reuse already defined variables.
            training: Whether in train or eval mode

        Returns:
            Output logits from the classifier
        """
        _, _, model = self.discriminator.discriminate(img, reuse=reuse, training=training, with_fc=False)
        model = self.discriminator.classify(model, num_classes, reuse=reuse, training=training)
        return model

    def discriminator(self, net, reuse=None, training=True):
        """Builds an encoder of the given inputs.

        Args:
            net: Input to the encoder (image)
            reuse: Whether to reuse already defined variables
            training: Whether in train or eval mode.

        Returns:
            Encoding of the input image.
        """
        f_dims = [32, 64, 128, 256, 384]
        with tf.variable_scope('disc2', reuse=reuse):
            with slim.arg_scope(toon_net_argscope(padding='SAME', training=training)):
                for l in range(0, self.num_layers):
                    net = slim.conv2d(net, num_outputs=f_dims[l], stride=1, scope='conv_{}'.format(l + 1))
                    net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope='pool_{}'.format(l + 1))
                net = slim.conv2d(net, 2, kernel_size=[1, 1], padding='VALID', activation_fn=None, normalizer_fn=None)
                enc_shape = net.get_shape().as_list()
                net = slim.avg_pool2d(net, kernel_size=enc_shape[1:3], stride=1)
                net = slim.flatten(net)
                return net

    def decoder(self, net, layer_id, reuse=None, training=True):
        """Builds a decoder on top of net.

        Args:
            net: Input to the decoder (output of encoder)
            reuse: Whether to reuse already defined variables
            training: Whether in train or eval mode.

        Returns:
            Decoded image with 3 channels.
        """
        f_dims = DEFAULT_FILTER_DIMS
        with tf.variable_scope('decoder', reuse=reuse):
            with slim.arg_scope(toon_net_argscope(padding='SAME', training=training, kernel_size=[4, 4])):
                layer = 0
                for l in range(0, self.num_layers):
                    net_shape = net.get_shape().as_list()
                    if net_shape[1]>self.im_shape[1]/4:
                        break
                    net = up_conv2d(net, num_outputs=f_dims[self.num_layers - l - layer_id - 1], scope='deconv_{}'.format(l))
                    layer += 1
                net = tf.image.resize_images(net, (self.im_shape[0], self.im_shape[1]))
                net = slim.conv2d(net, num_outputs=16, scope='deconv_{}'.format(layer), stride=1)
                net = slim.conv2d(net, num_outputs=3, scope='deconv_{}'.format(layer + 1), stride=1,
                                  activation_fn=tf.nn.tanh, normalizer_fn=None)
                return net

    def invertion_loss(self):
        # Define the losses for AE training
        ae_loss_scope = 'inversion_loss'
        ae_loss = tf.losses.mean_squared_error(self.enc_im, self.enc_dec_im, scope=ae_loss_scope, weights=100.0)
        tf.summary.scalar('losses/feature', ae_loss)
        disc_loss = tf.losses.softmax_cross_entropy(self.labels_fake(), self.disc_out, scope=ae_loss_scope,
                                                    weights=2.0)
        tv_loss = 5e-4*tf.reduce_mean(tf.image.total_variation(self.dec_im))
        tf.summary.scalar('losses/tv', tv_loss)
        losses_ae = tf.losses.get_losses(ae_loss_scope)
        losses_ae += tf.losses.get_regularization_losses(ae_loss_scope)
        losses_ae += [tv_loss]
        ae_total_loss = tf.add_n(losses_ae, name='ae_total_loss')
        return ae_total_loss

    def discriminator_loss(self):
        # Define loss for discriminator training
        disc_loss_scope = 'disc_loss'
        real_loss = tf.losses.softmax_cross_entropy(self.labels_real(), self.disc_out, scope=disc_loss_scope,
                                                    weights=1.0)
        tf.summary.scalar('losses/discriminator', real_loss)
        losses_disc = tf.losses.get_losses(disc_loss_scope)
        losses_disc += tf.losses.get_regularization_losses(disc_loss_scope)
        disc_total_loss = tf.add_n(losses_disc, name='disc_total_loss')

        real_pred = tf.arg_max(self.disc_out, 1)
        real_true = tf.arg_max(self.labels_real(), 1)
        tf.summary.scalar('accuracy/discriminator', slim.metrics.accuracy(real_pred, real_true))

        return disc_total_loss
