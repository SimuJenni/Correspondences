import tensorflow as tf

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops

import os
import sys
import numpy as np

from utils import montage_tf, get_variables_to_train, assign_from_checkpoint_fn, remove_missing, weights_montage
from constants import LOG_DIR
from Preprocessor_gray import Preprocessor

slim = tf.contrib.slim


class CNetTester:
    def __init__(self, model, dataset, pre_processor, tag='default'):
        tf.logging.set_verbosity(tf.logging.DEBUG)
        self.sess = tf.Session()
        self.graph = tf.Graph()
        self.model = model
        self.dataset = dataset
        self.tag = tag
        self.additional_info = None
        self.summaries = {}
        self.pre_processor_tiles = pre_processor
        self.pre_processor = Preprocessor((256, 256, 3))
        self.is_finetune = False
        self.num_eval_steps = None
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

    def get_log_dir(self):
        fname = '{}_{}_{}_test'.format(self.dataset.name, self.model.name, self.tag)
        return os.path.join(LOG_DIR, '{}/'.format(fname))

    def get_test_batch(self):
        with tf.device('/cpu:0'):
            # Get the training dataset
            train_set = self.dataset.get_testset()
            self.num_eval_steps = (self.dataset.get_num_test() / self.model.batch_size)
            print('Number of eval steps: {}'.format(self.num_eval_steps))
            provider = slim.dataset_data_provider.DatasetDataProvider(train_set, num_readers=1,
                                                                      common_queue_capacity=4 * self.model.batch_size,
                                                                      common_queue_min=2 * self.model.batch_size,
                                                                      )
            [img1] = provider.get(['image'])
            img2 = tf.identity(img1)

            # Pre-process data
            tiles_train = self.pre_processor_tiles.process_train(img1)
            img_train = self.pre_processor.process_train(img2)

            # Make batches
            [tiles_train, img_train] = tf.train.batch([tiles_train, img_train],
                                                      batch_size=self.model.batch_size,
                                                      num_threads=8,
                                                      capacity=5 * self.model.batch_size)
            print('tiles batch: {}'.format(tiles_train.get_shape().as_list()))

        batch_queue = slim.prefetch_queue.prefetch_queue([tiles_train, img_train], num_threads=4)
        return batch_queue.dequeue()

    def test(self, chpt_path=None):
        self.is_finetune = False
        with self.sess.as_default():
            with self.graph.as_default():
                tiles_train, imgs_train = self.get_test_batch()

                # Create the model
                pred = self.model.match(tiles_train, imgs_train, reuse=None, train=True)
                pred = tf.image.resize_nearest_neighbor(pred, (256, 256))
                pred_1, pred_2 = tf.split(pred, 2, axis=3)

                tiles = tf.unstack(tiles_train, axis=1)
                tile = tf.image.resize_bilinear(tiles[4], (256, 256))

                summary_ops = []
                summary_ops.append(tf.summary.image('pred/1', montage_tf(pred_1, 2, 8), max_outputs=1))
                summary_ops.append(tf.summary.image('pred/2', montage_tf(pred_2, 2, 8), max_outputs=1))
                summary_ops.append(tf.summary.image('input/img', montage_tf(imgs_train, 2, 8), max_outputs=1))
                summary_ops.append(tf.summary.image('input/tile', montage_tf(tile, 2, 8), max_outputs=1))

                # Start evaluation
                slim.evaluation.evaluation_loop('', self.get_save_dir(), logdir=self.get_log_dir(),
                                                num_evals=self.num_eval_steps,
                                                max_number_of_evaluations=10,
                                                summary_op=tf.summary.merge(summary_ops))