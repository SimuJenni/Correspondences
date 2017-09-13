import os

NUM_THREADS = 16

# Directories for data, images and models
DATA_DIR = '/Data/'
LOG_DIR = os.path.join(DATA_DIR, 'Logs/CNet/')

# Directories for tf-records
OBJECTNET3D_TF_DATADIR = os.path.join(DATA_DIR, 'TF_Records/ObjectNet3D-TFRecords/')
IMAGENET_TF_DATADIR = os.path.join(DATA_DIR, 'TF_Records/imagenet-CNET-TFRecords/')

PASCAL3D_TF_DATADIR = os.path.join(DATA_DIR, 'TF_Records/PASCAL3D-TFRecords/')
CELEBA_TF_DATADIR = os.path.join(DATA_DIR, 'TF_Records/CelebA-TFRecords/')

# Source directories for datasets
OBJECTNET3D_DATADIR = os.path.join(DATA_DIR, 'Datasets/ObjectNet3D/')
PASCAL3D_DATADIR = os.path.join(DATA_DIR, 'Datasets/PASCAL3D+_release1.1/')
CELEBA_DATADIR = os.path.join(DATA_DIR, 'Datasets/CelebA/')

IMAGENET_SRC_DIR = os.path.join(DATA_DIR, 'Datasets/ImageNet/')
IMAGENET_TRAIN_DIR = os.path.join(IMAGENET_SRC_DIR, 'ILSVRC2012_img_train/')
IMAGENET_VAL_DIR = os.path.join(IMAGENET_SRC_DIR, 'ILSVRC2012_img_val/')
