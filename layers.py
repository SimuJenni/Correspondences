import tensorflow as tf
from tensorflow.contrib import slim as slim
from tensorflow.python.ops import math_ops
import numpy as np


def my_dropout(x, keep_prob, scale=None, noise_shape=None, seed=None, name=None):
    with tf.name_scope(name, "dropout", [x]):
        x = tf.convert_to_tensor(x, name="x")
        keep_prob = tf.convert_to_tensor(keep_prob,
                                         dtype=x.dtype,
                                         name="keep_prob")

        noise_shape_gen = noise_shape if noise_shape is not None else tf.shape(x)
        if scale:
            noise_shape_gen = noise_shape_gen/[1, scale, scale, 1]

        # uniform [keep_prob, 1.0 + keep_prob)
        random_tensor = keep_prob
        random_tensor += tf.random_uniform(noise_shape_gen,
                                           seed=seed,
                                           dtype=x.dtype)
        if scale:
            random_tensor = tf.image.resize_nearest_neighbor(random_tensor, noise_shape[1:3])
        # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
        binary_tensor = math_ops.floor(random_tensor)
        ret = x * binary_tensor
        ret.set_shape(x.get_shape())
        return ret, binary_tensor


def random_select(a, b, keep_prob, batch_size):
    rand_vec = tf.random_uniform((batch_size,), minval=0.0, maxval=1.0)
    binary_tensor = tf.greater(rand_vec, keep_prob)
    return tf.where(binary_tensor, b, a), binary_tensor


def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)


def up_conv2d(net, num_outputs, scope, factor=2, resize_fun=tf.image.ResizeMethod.BILINEAR):
    in_shape = net.get_shape().as_list()
    net = tf.image.resize_images(net, (factor * in_shape[1], factor * in_shape[2]), resize_fun)
    net = slim.conv2d(net, num_outputs=num_outputs, scope=scope, stride=1)
    return net


def add_noise_plane(net, noise_channels, training=True):
    noise_shape = net.get_shape().as_list()
    noise_shape[-1] = noise_channels
    noise_planes = tf.random_normal(shape=noise_shape)
    biases = tf.Variable(tf.constant(0.0, shape=[noise_channels], dtype=tf.float32), trainable=True, name='noise_mu')
    if training:
        slim.add_model_variable(biases)
    noise_planes = tf.nn.bias_add(noise_planes, biases)
    return tf.concat(axis=3, values=[net, noise_planes], name='add_noise_{}'.format(noise_channels))


def sample(mu, log_var):
    noise_shape = mu.get_shape().as_list()
    noise = tf.random_normal(shape=noise_shape)
    samples = math_ops.add(math_ops.mul(math_ops.exp(log_var / 2.0), noise), mu)
    return samples


def ordered_merge(a, b, order):
    return tf.where(tf.python.math_ops.greater(order, 0), merge(a, b), merge(b, a))


def merge(a, b, dim=3):
    return tf.concat(axis=dim, values=[a, b])


def swap_merge(a, b):
    a1, a2 = tf.split(axis=0, num_or_size_splits=2, value=a)
    b1, b2 = tf.split(axis=0, num_or_size_splits=2, value=b)
    m1 = merge(a1, b1)
    m2 = merge(b2, a2)
    return merge(m1, m2, dim=0)


def spatial_shuffle(net, shuffle_prob):
    in_shape = net.get_shape().as_list()
    net = tf.transpose(net, [1, 2, 0, 3])
    net = tf.reshape(net, shape=[-1, in_shape[0], in_shape[3]])
    net_shuffled = tf.random_shuffle(net)
    net = tf.reshape(net, shape=[-1, in_shape[3]])
    net_shuffled = tf.reshape(net_shuffled, shape=[-1, in_shape[3]])
    net, binary_tensor = random_select(net, net_shuffled, shuffle_prob, net.get_shape().as_list()[0])
    net = tf.reshape(net, [in_shape[1], in_shape[2], in_shape[0], in_shape[3]])
    binary_tensor = tf.reshape(binary_tensor, [in_shape[1], in_shape[2], in_shape[0], 1])
    net = tf.transpose(net, [2, 0, 1, 3])
    binary_tensor = tf.transpose(binary_tensor, [2, 0, 1, 3])
    return net, tf.to_float(binary_tensor)


def pixel_dropout(net, p, kernel=None):
    input_shape = net.get_shape().as_list()
    noise_shape = np.array([input_shape[0], input_shape[1], input_shape[2], 1], dtype=np.int64)
    return my_dropout(net, p, kernel, noise_shape=noise_shape, name='pixel_dropout')


def spatial_drop_noise(net, kernel):
    input_shape = net.get_shape().as_list()
    noise_shape = np.array([input_shape[0], input_shape[1], input_shape[2], 1], dtype=np.int64)
    random_tensor = tf.random_uniform(noise_shape)
    random_tensor -= tf.reduce_min(random_tensor, [1, 2], keep_dims=True)
    #random_tensor = tf.pad(random_tensor, [[0, 0], [0, 1], [0, 1], [0, 0]])
    random_tensor = slim.max_pool2d(random_tensor, kernel, stride=1, padding='SAME')
    binary_tensor = math_ops.floor((random_tensor+1e-5)/tf.reduce_max(random_tensor, [1, 2], keep_dims=True))
    noise_tensor = tf.random_normal(net.get_shape()) * binary_tensor
    binary_tensor = tf.ones_like(binary_tensor) - binary_tensor
    ret = net * binary_tensor
    ret.set_shape(net.get_shape())
    return ret+noise_tensor, binary_tensor


def spatial_drop(net, kernel):
    input_shape = net.get_shape().as_list()
    noise_shape = np.array([input_shape[0], input_shape[1], input_shape[2], 1], dtype=np.int64)
    random_tensor = tf.random_uniform(noise_shape)
    random_tensor -= tf.reduce_min(random_tensor, [1, 2], keep_dims=True)
    #random_tensor = tf.pad(random_tensor, [[0, 0], [0, 1], [0, 1], [0, 0]])
    random_tensor = slim.max_pool2d(random_tensor, kernel, stride=1, padding='SAME')
    binary_tensor = math_ops.floor((random_tensor+1e-5)/tf.reduce_max(random_tensor, [1, 2], keep_dims=True))
    binary_tensor = tf.ones_like(binary_tensor) - binary_tensor
    ret = net * binary_tensor
    ret.set_shape(net.get_shape())
    return ret, binary_tensor


def channel_dropout(net, p):
    input_shape = net.get_shape().as_list()
    noise_shape = (input_shape[0], 1, 1, input_shape[3])
    return my_dropout(net, p, noise_shape=noise_shape, name='channel_dropout')


def coords2indices(coords, batch_size, size=32):
    center_idx = tf.concat(axis=1,
                           values=[tf.reshape(tf.range(batch_size), [batch_size, 1]), tf.to_int32(tf.squeeze(coords))])
    offsets = [[[0, i, j] for i in range(-size / 2, size / 2)] for j in range(-size / 2, size / 2)]
    patch_idx = [center_idx + o for o in offsets]
    return patch_idx


def conv_group(net, num_out, kernel_size, scope):
    input_groups = tf.split(axis=3, num_or_size_splits=2, value=net)
    output_groups = [slim.conv2d(j, num_out / 2, kernel_size=kernel_size, scope='{}_{}'.format(scope, idx))
                     for (idx, j) in enumerate(input_groups)]
    return tf.concat(axis=3, values=output_groups)


def res_block_bottleneck(inputs, depth, depth_bottleneck, noise_channels=None, scope=None):
    with slim.variable_scope.variable_scope(scope):
        shortcut = inputs
        preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')
        if noise_channels:
            preact = add_noise_plane(preact, noise_channels)
        residual = slim.conv2d(preact, depth_bottleneck, kernel_size=[1, 1], stride=1, scope='conv1')
        residual = slim.conv2d(residual, depth_bottleneck, kernel_size=[3, 3], stride=1, scope='conv2')
        residual = slim.conv2d(residual, depth, kernel_size=[1, 1], stride=1, normalizer_fn=None, activation_fn=None,
                               scope='conv3')
        output = shortcut + residual
        return output


def maxpool2d_valid(inputs, kernel_size, stride, scope=None):
    pad_total = kernel_size[0] - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
    return slim.max_pool2d(inputs, kernel_size, stride=stride, padding='VALID', scope=scope)
