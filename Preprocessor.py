from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


class Preprocessor:
    def __init__(self, target_shape, augment_color=False, aspect_ratio_range=(0.8, 1.2), area_range=(0.333, 1.0)):
        self.target_shape = target_shape
        self.augment_color = augment_color
        self.aspect_ratio_range = aspect_ratio_range
        self.area_range = area_range

    def central_crop(self, image):
        # Crop the central region of the image with an area containing 85% of the original image.
        image = tf.image.central_crop(image, central_fraction=0.85)

        # Resize the image to the original height and width.
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bilinear(image, [self.target_shape[0], self.target_shape[1]], align_corners=False)
        image = tf.squeeze(image, [0])

        # Resize to output size
        image.set_shape([self.target_shape[0], self.target_shape[1], 3])
        return image

    def extract_random_patch(self, image):
        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
            tf.shape(image),
            [[[0, 0, 1, 1]]],
            aspect_ratio_range=self.aspect_ratio_range,
            area_range=self.area_range,
            use_image_if_no_bounding_boxes=True)
        bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box

        # Crop the image to the specified bounding box.
        image = tf.slice(image, bbox_begin, bbox_size)
        image = tf.expand_dims(image, 0)
        resized_image = tf.cond(
            tf.random_uniform(shape=(), minval=0.0, maxval=1.0) > 0.5,
            fn1=lambda: tf.image.resize_bilinear(image, self.target_shape[:2], align_corners=False),
            fn2=lambda: tf.image.resize_bicubic(image, self.target_shape[:2], align_corners=False))
        image = tf.squeeze(resized_image)
        image.set_shape(self.target_shape)
        return image

    def color_augment_and_scale(self, image):
        image = tf.to_float(image) / 255.

        if self.augment_color:
            bright_delta, sat, hue_delta, cont = sample_color_params()
            image = dist_color(image, bright_delta, sat, hue_delta, cont)
            image = tf.clip_by_value(image, 0.0, 1.0)

        # Scale to [-1, 1]
        image = tf.to_float(image) * 2. - 1.
        return image

    def process_train(self, image):
        image = self.extract_random_patch(image)
        image = self.color_augment_and_scale(image)
        image = tf.image.random_flip_left_right(image)
        return image

    def process_test(self, image):
        image = self.central_crop(image)
        image = self.color_augment_and_scale(image)
        image = tf.image.random_flip_left_right(image)
        return image


class VOCPreprocessor(Preprocessor):
    def __init__(self, target_shape, augment_color=True, aspect_ratio_range=(0.9, 1.1), area_range=(0.1, 1.0)):
        Preprocessor.__init__(self, target_shape, augment_color, aspect_ratio_range, area_range)

    def process_test(self, image):
        image = self.extract_random_patch(image)
        image = self.color_augment_and_scale(image)
        image = tf.image.random_flip_left_right(image)
        return image


def sample_color_params(bright_max_delta=16./255., lower_sat=0.7, upper_sat=1.3, hue_max_delta=0.05, lower_cont=0.7,
                        upper_cont=1.3):
    bright_delta = tf.random_uniform([], -bright_max_delta, bright_max_delta)
    sat = tf.random_uniform([], lower_sat, upper_sat)
    hue_delta = tf.random_uniform([], -hue_max_delta, hue_max_delta)
    cont = tf.random_uniform([], lower_cont, upper_cont)
    return bright_delta, sat, hue_delta, cont


def dist_color(image, bright_delta, sat, hue_delta, cont):
    image1 = tf.image.adjust_brightness(image, delta=bright_delta)
    image1 = tf.image.adjust_saturation(image1, saturation_factor=sat)
    image1 = tf.image.adjust_hue(image1, delta=hue_delta)
    image1 = tf.image.adjust_contrast(image1, contrast_factor=cont)

    image2 = tf.image.adjust_brightness(image, delta=bright_delta)
    image2 = tf.image.adjust_contrast(image2, contrast_factor=cont)
    image2 = tf.image.adjust_saturation(image2, saturation_factor=sat)
    image2 = tf.image.adjust_hue(image2, delta=hue_delta)

    aug_image = tf.cond(tf.random_uniform(shape=(), minval=0.0, maxval=1.0) > 0.5,
                        fn1=lambda: image1, fn2=lambda: image2)
    return aug_image


def flip_lr(image, p):
    return tf.cond(p > 0.5, fn1=lambda: image, fn2=lambda: tf.image.flip_left_right(image))
