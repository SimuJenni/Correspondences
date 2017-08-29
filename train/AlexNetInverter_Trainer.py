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
        self.num_train_steps = None
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

    def get_train_batch(self, dataset_id):
        with tf.device('/cpu:0'):
            # Get the training dataset
            if dataset_id:
                train_set = self.dataset.get_split(dataset_id)
                self.num_train_steps = (self.dataset.get_num_dataset(
                    dataset_id) / self.model.batch_size) * self.num_epochs
            else:
                train_set = self.dataset.get_trainset()
                self.num_train_steps = (self.dataset.get_num_train() / self.model.batch_size) * self.num_epochs
            print('Number of training steps: {}'.format(self.num_train_steps))
            provider = slim.dataset_data_provider.DatasetDataProvider(train_set, num_readers=4,
                                                                      common_queue_capacity=20 * self.model.batch_size,
                                                                      common_queue_min=10 * self.model.batch_size)

            # Parse a serialized Example proto to extract the image and metadata.
            [img_train] = provider.get(['image'])

            # Pre-process data
            img_train = self.pre_processor.process_train(img_train)

            # Make batches
            imgs_train = tf.train.batch([img_train],
                                        batch_size=self.model.batch_size,
                                        num_threads=8,
                                        capacity=5 * self.model.batch_size)
            batch_queue = slim.prefetch_queue.prefetch_queue([imgs_train])
            return batch_queue.dequeue()

    def classification_loss(self, preds_train, labels_train):
        # Define the loss
        loss_scope = 'classification_loss'
        if self.dataset.is_multilabel:
            train_loss = tf.contrib.losses.sigmoid_cross_entropy(preds_train, labels_train, scope=loss_scope)
        else:
            train_loss = tf.contrib.losses.softmax_cross_entropy(preds_train, labels_train, scope=loss_scope)
        tf.summary.scalar('losses/training loss', train_loss)
        train_losses = tf.losses.get_losses(loss_scope)
        train_losses += tf.losses.get_regularization_losses(loss_scope)
        total_train_loss = math_ops.add_n(train_losses, name='total_train_loss')

        # Compute accuracy
        if not self.dataset.is_multilabel:
            predictions = tf.argmax(preds_train, 1)
            tf.summary.scalar('accuracy/training accuracy',
                              slim.metrics.accuracy(predictions, tf.argmax(labels_train, 1)))
            tf.summary.histogram('labels', tf.argmax(labels_train, 1))
            tf.summary.histogram('predictions', predictions)
        return total_train_loss

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
        num_train_steps = self.num_train_steps
        boundaries = [np.int64(num_train_steps * 0.2), np.int64(num_train_steps * 0.4),
                      np.int64(num_train_steps * 0.6), np.int64(num_train_steps * 0.8)]
        values = [0.01, 0.01 * 250. ** (-1. / 4.), 0.01 * 250 ** (-2. / 4.), 0.01 * 250 ** (-3. / 4.),
                  0.01 * 250. ** (-1.)]
        return tf.train.piecewise_constant(self.global_step, boundaries=boundaries, values=values)

    def learning_rate_linear(self):
        return tf.train.polynomial_decay(self.init_lr, self.global_step, 0.9 * self.num_train_steps,
                                         end_learning_rate=self.end_lr)

    def make_init_fn(self, chpt_path):
        var2restore = slim.get_variables_to_restore(include=['discriminator'])
        init_fn = assign_from_checkpoint_fn(chpt_path, var2restore, ignore_missing_vars=True)
        print('Variables to restore: {}'.format([v.op.name for v in var2restore]))
        sys.stdout.flush()
        return init_fn

    def train_inverter(self, chpt_path, dataset_id=None):
        print('Restoring from: {}'.format(chpt_path))
        self.is_finetune = True
        with self.sess.as_default():
            with self.graph.as_default():
                # Get training batches
                imgs_train = self.get_train_batch(dataset_id)

                # Get predictions
                inv_im = self.model.net(imgs_train)

                # Compute the loss
                disc_loss = self.model.discriminator_loss()
                invertion_loss = self.model.invertion_loss()

                # Handle dependencies
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                if update_ops:
                    updates = tf.group(*update_ops)
                    disc_loss = control_flow_ops.with_dependencies([updates], disc_loss)
                    invertion_loss = control_flow_ops.with_dependencies([updates], invertion_loss)

                # Make summaries
                tf.summary.image('imgs/inv_imgs', montage_tf(inv_im, 2, 8), max_outputs=1)
                tf.summary.image('imgs/imgs', montage_tf(imgs_train, 2, 8), max_outputs=1)
                self.make_summaries()

                # Create training operation
                train_op_disc = self.make_train_op(disc_loss, self.optimizer(), scope='disc2')
                train_op_dec = self.make_train_op(invertion_loss, self.optimizer(), scope='decoder')


                # Start training
                slim.learning.train(train_op_disc + train_op_dec, self.get_save_dir(),
                                    init_fn=self.make_init_fn(chpt_path),
                                    number_of_steps=self.num_train_steps,
                                    save_summaries_secs=300, save_interval_secs=3000,
                                    log_every_n_steps=100)
