import os
import tensorflow as tf

from Preprocessor import Preprocessor
from train.AlexNet_NN_search_full import CNetTrainer
from datasets.ImageNet import ImageNet
from models.AlexNet_layers_lrelu import AlexNet

from constants import IMAGENET_VAL_DIR
from scipy import misc
import cv2


def load_val_image(class_id, val_dir=IMAGENET_VAL_DIR):
    class_folders = os.listdir(val_dir)
    img_names = os.listdir(os.path.join(IMAGENET_VAL_DIR, class_folders[class_id]))
    img = misc.imread(os.path.join(IMAGENET_VAL_DIR, class_folders[class_id], img_names[0]), mode='RGB')
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
    return img


target_shape = [227, 227, 3]
model = AlexNet(batch_size=2000)
data = ImageNet()
preprocessor = Preprocessor(target_shape=target_shape)
ckpt = '/Data/Logs/CNet/imagenet_SDNet_res1_default_baseline_finetune_conv_5/model.ckpt-324174'
#ckpt = '/Data/Logs/CNet/imagenet_AlexNet_sorted_alex_sorted_finetune_conv_4/model.ckpt-450360'
#ckpt = '/Data/Logs/CNet/imagenet_AlexNet_sorted2_alex_sorted_finetune_conv_5/model.ckpt-294132'

trainer = CNetTrainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=1, tag='inv_tv',
                      lr_policy='linear', optimizer='adam', init_lr=0.0003, end_lr=0.00003)
# trainer.compute_stats(ckpt, 4, model.name)


# imgs: 0, 3, 15, 26, 87, 95, 98, 146, 221, 229, 237, 259, 348, 378, 388, 422
# for i in range(87, 1000):
#     print(i)
#     img = load_val_image(i)
#     misc.imshow(img)


for i in [26]:
    trainer = CNetTrainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=1, tag='inv_tv',
                         lr_policy='linear', optimizer='adam', init_lr=0.0003, end_lr=0.00003)
    trainer.search_nn(load_val_image(i), ckpt, 4, model.name, i)

