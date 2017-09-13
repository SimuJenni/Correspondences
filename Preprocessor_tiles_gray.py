import tensorflow as tf

slim = tf.contrib.slim


class Preprocessor:
    def __init__(self, target_shape, crop_size, augment_color=False, aspect_ratio_range=(0.7, 1.3), area_range=(0.5, 1.0)):
        self.tile_shape = target_shape
        self.crop_size = crop_size
        self.augment_color = augment_color
        self.aspect_ratio_range = aspect_ratio_range
        self.area_range = area_range

    def crop_3x3_tile_block(self, image, bbox=(0., 0., 1., 1.)):
        crop_size = [self.crop_size[0], self.crop_size[1], 1]
        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
            tf.shape(image),
            [[bbox]],
            aspect_ratio_range=self.aspect_ratio_range,
            area_range=(0.8, 1.0),
            use_image_if_no_bounding_boxes=True,
            min_object_covered=0.5)
        bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box

        # Crop the image to the specified bounding box.
        image = tf.slice(image, bbox_begin, bbox_size)
        image = tf.expand_dims(image, 0)

        resized_image = tf.cond(
            tf.random_uniform(shape=(), minval=0.0, maxval=1.0) > 0.5,
            true_fn=lambda: tf.image.resize_bilinear(image, self.crop_size, align_corners=False),
            false_fn=lambda: tf.image.resize_bicubic(image, self.crop_size, align_corners=False))
        image = tf.squeeze(resized_image, axis=0)
        image.set_shape(crop_size)
        return image

    def extract_tiles(self, image):
        tiles = []
        dx = self.crop_size[0]/3
        dy = self.crop_size[1]/3
        for x in range(3):
            for y in range(3):
                tile = tf.slice(image, [dx*x, dy*y, 0], [dx, dy, 1])
                tile = self.extract_random_patch(tile)
                tiles.append(tile)
        return tiles

    def central_crop(self, image):
        # Crop the central region of the image with an area containing 85% of the original image.
        image = tf.image.central_crop(image, central_fraction=0.85)

        # Resize the image to the original height and width.
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bilinear(image, [self.tile_shape[0], self.tile_shape[1]], align_corners=False)
        image = tf.squeeze(image, [0])

        # Resize to output size
        image.set_shape([self.tile_shape[0], self.tile_shape[1], 3])
        return image

    def extract_random_patch(self, image, bbox=(0., 0., 1., 1.)):
        image = tf.cond(
            tf.random_uniform(shape=(), minval=0.0, maxval=1.0) > 0.5,
            true_fn=lambda: tf.contrib.image.rotate(image, tf.random_uniform((1,), minval=-0.2, maxval=0.2),
                                                    interpolation='BILINEAR'),
            false_fn=lambda: image)

        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
            tf.shape(image),
            [[bbox]],
            aspect_ratio_range=self.aspect_ratio_range,
            area_range=self.area_range,
            use_image_if_no_bounding_boxes=True,
            min_object_covered=0.01)
        bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box

        # Crop the image to the specified bounding box.
        image = tf.slice(image, bbox_begin, bbox_size)
        image = tf.expand_dims(image, 0)

        resized_image = tf.cond(
            tf.random_uniform(shape=(), minval=0.0, maxval=1.0) > 0.5,
            true_fn=lambda: tf.image.resize_bilinear(image, self.tile_shape[:2], align_corners=False),
            false_fn=lambda: tf.image.resize_bicubic(image, self.tile_shape[:2], align_corners=False))
        image = tf.squeeze(resized_image, axis=0)
        image.set_shape(self.tile_shape)
        return image

    def color_augment_and_scale(self, image):
        image = tf.to_float(image) / 255.
        image = tf.image.rgb_to_grayscale(image)
        image = tf.to_float(image) * 2. - 1.
        return image

    def process_train(self, image):
        image = self.color_augment_and_scale(image)
        tile_block = self.crop_3x3_tile_block(image)
        tile_block = tf.image.random_flip_left_right(tile_block)
        print('img block: {}'.format(tile_block.get_shape().as_list()))
        tf.summary.image('input/block', tf.expand_dims(tile_block, 0), max_outputs=1)
        tiles = self.extract_tiles(tile_block)
        print('tile {}'.format(tiles[0].get_shape().as_list()))
        for i, tile in enumerate(tiles):
            tf.summary.image('imgs/tile_{}'.format(i), tf.expand_dims(tile, 0), max_outputs=1)

        tiles = tf.stack(tiles)
        print('tiles: {}'.format(tiles.get_shape().as_list()))
        return tiles

    def process_test(self, image):
        image = self.central_crop(image)
        image = self.color_augment_and_scale(image)
        image = tf.image.random_flip_left_right(image)
        return image
