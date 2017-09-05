import tensorflow as tf

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
import matplotlib.pyplot as plt


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
        self.model = model
        self.dataset = dataset
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
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
            provider = slim.dataset_data_provider.DatasetDataProvider(test_set, num_readers=1, shuffle=False)

            # Parse a serialized Example proto to extract the image and metadata.
            [img_test] = provider.get(['image'])

            # Pre-process data
            img_test = self.pre_processor.process_test(img_test)

            # Make batches
            imgs_test = tf.train.batch([img_test],
                                        batch_size=self.model.batch_size,
                                        num_threads=1,
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

    def search_nn(self, query_img, chpt_path, layer_id, model_name, im_id):
        print('Restoring from: {}'.format(chpt_path))
        self.is_finetune = True
        mu = np.load('mu_flat_{}_{}.npy'.format(model_name, layer_id))

        with self.sess.as_default():
            with self.graph.as_default():

                imgs_tf = tf.placeholder(tf.uint8, shape=query_img.shape, name='imgs_tf')
                imgs_tf_ = tf.expand_dims(self.pre_processor.process_test(imgs_tf), axis=0)
                _, enc_q = self.model.classifier(imgs_tf_, self.dataset.num_classes, training=False, reuse=None)
                enc_l = self.pool_and_norm(enc_q[layer_id], mu)

                vars = slim.get_variables_to_restore()
                print('Variables to restore: {}'.format([v.op.name for v in vars]))
                saver = tf.train.Saver(var_list=vars)
                saver.restore(self.sess, chpt_path)
                target_enc = self.sess.run(enc_l, feed_dict={imgs_tf: query_img})

                target_tf = tf.placeholder(tf.float32, shape=target_enc.shape, name='target_tf')
                imgs_train = self.get_test_batch()
                _, enc_layers = self.model.classifier(imgs_train, self.dataset.num_classes, training=False, reuse=True)
                enc_ls = self.pool_and_norm(enc_layers[layer_id], mu)
                print('enc_ls shape: {}'.format(enc_ls.get_shape().as_list()))

                dist = tf.reduce_sum(enc_ls*target_tf, axis=1)
                # dist = tf.norm(enc_ls-target_enc, axis=1)
                ds, inds = tf.nn.top_k(dist, k=10)
                nn_imgs = tf.gather(imgs_train, inds)

                sv = tf.train.Supervisor()
                sv.start_queue_runners(self.sess)
                f, axarr = plt.subplots(1, 11, figsize=(25,5))
                plt.ion()
                plt.show(block=False)

                axarr[0].imshow(query_img)
                best_d = -1.*np.inf * np.ones(ds.get_shape().as_list())
                best_imgs = np.zeros(nn_imgs.get_shape().as_list())

                for i in range(self.num_test_steps):
                    [d, img] = self.sess.run([ds, nn_imgs], feed_dict={target_tf: target_enc})
                    tmp_d = np.append(d, best_d, axis=0)
                    tmp_img = np.append(img, best_imgs, axis=0)

                    best_inds = tmp_d.argsort()[-10:][::-1]
                    best_d = tmp_d[best_inds]
                    best_imgs = tmp_img[best_inds]

                    for j in range(10):
                        axarr[j+1].imshow((np.squeeze(best_imgs[j])+1.0)/2.0)
                    f.canvas.draw()

                    print(best_d)

                f.savefig('{}_{}_{}.png'.format(model_name, layer_id, im_id))

    def compute_stats(self, chpt_path, layer_id, model_name):
        print('Restoring from: {}'.format(chpt_path))
        self.is_finetune = True
        with self.sess.as_default() as sess:
            with self.graph.as_default():
                imgs_train = self.get_test_batch()
                _, enc_layers = self.model.classifier(imgs_train, self.dataset.num_classes, training=False, reuse=None)
                enc_ls = self.hybrid_pool(enc_layers[layer_id])
                print('enc_ls shape: {}'.format(enc_ls.get_shape().as_list()))

                vars = slim.get_variables_to_restore(include=['discriminator'])
                print('Variables to restore: {}'.format([v.op.name for v in vars]))
                self.sess.run(tf.global_variables_initializer())

                saver = tf.train.Saver(var_list=vars)
                saver.restore(self.sess, chpt_path)
                mu, _ = tf.nn.moments(enc_ls, axes=[0])

                sv = tf.train.Supervisor()
                sv.start_queue_runners(sess)

                mu_ = np.zeros(mu.get_shape().as_list())

                for i in range(self.num_test_steps):
                    print('Iteration {}/{}'.format(i, self.num_test_steps))
                    [mu_i] = self.sess.run([mu])
                    mu_ += mu_i

                mu_final = mu_/self.num_test_steps
                np.save('mu_flat_{}_{}'.format(model_name, layer_id), mu_final)

    def pool_and_norm(self, net, mu):
        net = self.hybrid_pool(net)
        net = net-mu
        net = tf.nn.l2_normalize(net, dim=1)
        return net

    def hybrid_pool(self, net):
        net = slim.flatten(net)
        return net


