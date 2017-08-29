import tensorflow as tf

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops

import os
import sys
import numpy as np

from utils import montage_tf, get_variables_to_train, assign_from_checkpoint_fn, remove_missing, weights_montage
from constants import LOG_DIR

slim = tf.contrib.slim


class CNetTrainer:
    def __init__(self, model, dataset, pre_processor, num_epochs, optimizer='adam', lr_policy='const', init_lr=0.0003,
                 tag='default', end_lr=None, reinit_fc=False):
        tf.logging.set_verbosity(tf.logging.DEBUG)
        self.sess = tf.Session()
        self.graph = tf.Graph()
        self.model = model
        self.dataset = dataset
        self.num_epochs = num_epochs
        self.tag = tag
        self.additional_info = None
        self.summaries = {}
        self.pre_processor = pre_processor
        self.opt_type = optimizer
        self.lr_policy = lr_policy
        self.init_lr = init_lr
        self.end_lr = end_lr if end_lr is not None else 0.01 * init_lr
        self.is_finetune = False
        self.num_test_steps = None
        self.reinit_fc = reinit_fc
        self.opt_g = None
        self.opt_d = None
        with self.sess.as_default():
            with self.graph.as_default():
                self.global_step = slim.create_global_step()

    def get_save_dir(self):
        fname = '{}_{}_{}'.format(self.dataset.name, self.model.name, self.tag)
        if self.is_finetune:
            fname = '{}_finetune'.format(fname)
        if self.additional_info:
            fname = '{}_{}'.format(fname, self.additional_info)
        return os.path.join(LOG_DIR, '{}/'.format(fname))

    def optimizer(self):
        opts = {'adam': tf.train.AdamOptimizer(learning_rate=self.learning_rate(), beta1=0.9, epsilon=1e-5),
                'sgd': tf.train.MomentumOptimizer(learning_rate=self.learning_rate(), momentum=0.9)}
        return opts[self.opt_type]

    def learning_rate(self):
        policies = {'const': self.init_lr,
                    'alex': self.learning_rate_alex(),
                    'linear': self.learning_rate_linear()}
        return policies[self.lr_policy]

    def get_test_batch(self):
        with tf.device('/cpu:0'):
            # Get the training dataset
            test_set = self.dataset.get_testset()
            self.num_test_steps = (self.dataset.get_num_test() / self.model.batch_size) * self.num_epochs
            print('Number of training steps: {}'.format(self.num_test_steps))
            provider = slim.dataset_data_provider.DatasetDataProvider(test_set, num_readers=1, shuffle=False,
                                                                      common_queue_capacity=20 * self.model.batch_size,
                                                                      common_queue_min=10 * self.model.batch_size)

            # Parse a serialized Example proto to extract the image and metadata.
            [img_test] = provider.get(['image'])

            # Pre-process data
            img_test = self.pre_processor.process_test(img_test)

            # Make batches
            imgs_test = tf.train.batch([img_test],
                                        batch_size=self.model.batch_size,
                                        num_threads=8,
                                        capacity=5 * self.model.batch_size)
            batch_queue = slim.prefetch_queue.prefetch_queue([imgs_test])
            return batch_queue.dequeue()

    def make_train_op(self, loss, optimizer, vars2train=None, scope=None):
        if scope:
            vars2train = get_variables_to_train(trainable_scopes=scope)
        train_op = slim.learning.create_train_op(loss, optimizer, variables_to_train=vars2train,
                                                 global_step=self.global_step, summarize_gradients=True)
        return train_op

    def make_summaries(self):
        # Handle summaries
        for variable in slim.get_model_variables():
            tf.summary.histogram(variable.op.name, variable)

    def learning_rate_alex(self):
        # Define learning rate schedule
        num_train_steps = self.num_test_steps
        boundaries = [np.int64(num_train_steps * 0.2), np.int64(num_train_steps * 0.4),
                      np.int64(num_train_steps * 0.6), np.int64(num_train_steps * 0.8)]
        values = [0.01, 0.01 * 250. ** (-1. / 4.), 0.01 * 250 ** (-2. / 4.), 0.01 * 250 ** (-3. / 4.),
                  0.01 * 250. ** (-1.)]
        return tf.train.piecewise_constant(self.global_step, boundaries=boundaries, values=values)

    def learning_rate_linear(self):
        return tf.train.polynomial_decay(self.init_lr, self.global_step, 0.9 * self.num_test_steps,
                                         end_learning_rate=self.end_lr)

    def make_init_fn(self, chpt_path):
        var2restore = slim.get_variables_to_restore(include=['discriminator'])
        init_fn = assign_from_checkpoint_fn(chpt_path, var2restore, ignore_missing_vars=True)
        print('Variables to restore: {}'.format([v.op.name for v in var2restore]))
        sys.stdout.flush()
        return init_fn

    def search_nn(self, query_img, chpt_path, layer_id):
        print('Restoring from: {}'.format(chpt_path))
        self.is_finetune = True
        with self.sess.as_default():
            imgs_tf = tf.placeholder(tf.uint8, shape=[self.model.batch_size, 224, 224, 3], name='imgs_tf')
            imgs_tf = tf.to_float(imgs_tf) / 127.5 - 1.0
            _, enc_q = self.model.classifier(imgs_tf, self.dataset.num_classes, training=False, reuse=None)
            t_enc = slim.flatten(enc_q[layer_id])

            vars = slim.get_variables_to_restore(include=['discriminator'])
            print('Variables to restore: {}'.format([v.op.name for v in vars]))
            saver = tf.train.Saver(var_list=vars)
            saver.restore(self.sess, chpt_path)
            self.sess.run(tf.global_variables_initializer())
            target_enc = self.sess.run([t_enc], feed_dict={imgs_tf: query_img})

            target_tf = tf.placeholder(tf.float32, shape=target_enc.shape, name='target_tf')
            imgs_train = self.get_test_batch()
            _, enc_l = self.model.classifier(imgs_train, self.dataset.num_classes, training=False, reuse=True)
            enc_imgs = slim.flatten(enc_l[layer_id])
            dist = tf.norm(enc_imgs-target_tf)
            for i in self.num_test_steps:
                d = self.sess.run(dist, feed_dict={target_tf: target_enc})
                print(d)



