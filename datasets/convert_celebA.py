from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import scipy.io

from scipy import misc

import tensorflow as tf
from constants import PASCAL3D_DATADIR, PASCAL3D_TF_DATADIR

from datasets import dataset_utils


def parse_annotation(data_path):
    txt_file = os.path.join(data_path, 'Anno/list_landmarks_align_celeba.txt')

    with tf.Graph().as_default():
        coder = dataset_utils.ImageCoder()
        with tf.Session('') as sess:
            with open(txt_file) as file:
                for i, line in enumerate(file):
                    fields = line.split()
                    if len(fields) < 11:
                        continue
                    image_name = fields[0]
                    image_path = os.path.join(data_path, 'Img/img_align_celeba/'.format(image_name))
                    img = misc.imread(image_path, mode='RGB')

                    coords = fields[1:]
                    x_coords = coords[0:-1:2]
                    y_coords = coords[1:-1:2] + [coords[-1]]
                    #TODO ...

    examples = []

    mat = scipy.io.loadmat(mat_file)
    record = mat['record']

    # Get the filename
    fname = record['filename'][0, 0][0]
    image_path = os.path.join(data_path, 'Images', '{}_imagenet'.format(class_name), fname)

    # Get the image size
    im_size = record['imgsize'][0, 0][0]

    # Get all the objects
    objects = mat['record']['objects'][0, 0]
    num_objects = objects.shape[1]
    for i in range(num_objects):
        class_name = objects[0, i]['class'][0]
        bbox = objects[0, i]['bbox'][0]

        # Get the viewpoint information
        viewpoint = objects[0, i]['viewpoint'][0, 0]
        azimuth = viewpoint['azimuth_coarse'][0, 0]
        elevation = viewpoint['elevation_coarse'][0, 0]
        theta = viewpoint['theta'][0, 0]

        difficult = objects[0, i]['difficult'][0, 0]
        occluded = objects[0, i]['occluded'][0, 0]
        truncated = objects[0, i]['truncated'][0, 0]

        if difficult + occluded + truncated == 0:
            examples.append({'image_path': image_path, 'im_size': im_size, 'class_name': class_name, 'bbox': bbox,
                             'azimuth': azimuth, 'elevation': elevation, 'theta': theta})
    return examples


def to_tfrecord(image_ids_file, dest_dir, source_dir, class_name):
    if not tf.gfile.Exists(dest_dir):
        tf.gfile.MakeDirs(dest_dir)
    with open(image_ids_file) as f:
        img_ids = f.readlines()

    num_images = len(img_ids)
    num_stored = 0
    writer = tf.python_io.TFRecordWriter(get_output_filename(dest_dir, class_name))

    with tf.Graph().as_default():
        coder = dataset_utils.ImageCoder()

        with tf.Session('') as sess:
            for j in range(num_images):
                # Parse the annotations file
                mat_path = os.path.join(source_dir, 'Annotations',
                                        '{}_imagenet/{}.mat'.format(class_name, img_ids[j].strip('\n')))
                examples = parse_mat(mat_path, source_dir, class_name)

                for e in examples:
                    sys.stdout.write('\r>> Reading file [%s] image %d/%d' % (e['image_path'], j + 1, num_images))
                    sys.stdout.flush()

                    # Get image, edge-map and cartooned image
                    img = misc.imread(e['image_path'], mode='RGB')

                    # Encode the images
                    image_str = coder.encode_jpeg(img)

                    # Build example
                    example = dataset_utils.to_tfexample(image_str, 'jpg', e['im_size'].tolist(), e['bbox'].tolist(),
                                                         e['azimuth'], e['elevation'], e['theta'])
                    # Write example
                    writer.write(example.SerializeToString())

                    # Update number of examples per class
                    num_stored += 1
    return num_stored


def get_output_filename(dest_dir, class_name):
    """Creates the output filename.
    Args:
      dataset_dir: The dataset directory where the dataset is stored.
      split_name: The name of the train/test split.
    Returns:
      An absolute file path.
    """
    return '%s/PASCAL3D_%s.tfrecord' % (dest_dir, class_name)


def run(target_dir=PASCAL3D_TF_DATADIR, source_dir=PASCAL3D_DATADIR):
    """Runs the conversion operation.
    Args:
      target_dir: The dataset directory where the dataset is stored.
    """
    if not tf.gfile.Exists(target_dir):
        tf.gfile.MakeDirs(target_dir)

    num_per_class = {cn: 0 for cn in CLASSES}
    dataset_utils.save_obj(num_per_class, target_dir, 'num_per_class')
    num_per_class = dataset_utils.load_obj('num_per_class', target_dir)

    for c in CLASSES:
        train_dir = os.path.join(target_dir, '{}_train'.format(c))
        val_dir = os.path.join(target_dir, '{}_val'.format(c))

        # Process the train data:
        filename = os.path.join(source_dir, 'Image_sets', '{}_imagenet_train.txt'.format(c))
        num_per_class['{}_train'.format(c)] = to_tfrecord(filename, train_dir, source_dir, c)

        # Process the val data:
        filename = os.path.join(source_dir, 'Image_sets', '{}_imagenet_val.txt'.format(c))
        num_per_class['{}_val'.format(c)] = to_tfrecord(filename, val_dir, source_dir, c)

    print('\nFinished converting the Pascal3D dataset!')
    dataset_utils.save_obj(num_per_class, target_dir, 'num_per_class')


if __name__ == '__main__':
    run()
