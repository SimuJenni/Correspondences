import os

from Preprocessor import Preprocessor
from train.AlexNet_NN_search import CNetTrainer
from datasets.ImageNet import ImageNet
from models.AlexNet_layers import AlexNet
from constants import IMAGENET_VAL_DIR
from scipy import misc


def load_val_images(num_imgs, val_dir=IMAGENET_VAL_DIR, num_per_class=1):
    class_folders = os.listdir(val_dir)
    imgs = []
    for i, dir_name in enumerate(class_folders):
        if i > num_imgs-1:
            break
        img_names = os.listdir(os.path.join(IMAGENET_VAL_DIR, dir_name))
        img = misc.imread(os.path.join(IMAGENET_VAL_DIR, dir_name, img_names[0]), mode='RGB')
        img = misc.imresize(img, [256, 256, 3])
        return img
        imgs.append(img)

    imgs_stacked = np.stack(imgs, 0)
    return imgs_stacked


target_shape = [224, 224, 3]
model = AlexNet(batch_size=2048)
data = ImageNet()
preprocessor = Preprocessor(target_shape=target_shape)
trainer = CNetTrainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=10, tag='inv_tv',
                      lr_policy='linear', optimizer='adam', init_lr=0.0003, end_lr=0.00003)
ckpt = '/Data/Logs/CNet/imagenet_SDNet_res1_default_baseline_finetune_conv_5/model.ckpt-324174'

img = load_val_images(1)
#trainer.search_nn(load_val_images(1), ckpt, 1)
trainer.compute_stats(load_val_images(1), ckpt, 4)
