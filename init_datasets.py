from datasets import convert_imagenet

from constants import *

# download_and_convert_stl10.run()
# convert_voc2007.run(VOC2007_TF_DATADIR, VOC2007_SRC_DIR)
convert_imagenet.run(IMAGENET_TRAIN_DIR, IMAGENET_VAL_DIR, IMAGENET_TF_DATADIR)
