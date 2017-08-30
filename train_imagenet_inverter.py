from Preprocessor import Preprocessor
from train.AlexNetInverter_Trainer import CNetTrainer
from datasets.ImageNet import ImageNet
from models.AlexNetInverter import AlexNetInverter
from models.AlexNet_chan_sort import AlexNet
from constants import LOG_DIR
import os

target_shape = [128, 128, 3]
alexnet = AlexNet(32, pad='SAME')
model = AlexNetInverter(model=alexnet, target_shape=target_shape, layer_id=0)
data = ImageNet()
preprocessor = Preprocessor(target_shape=target_shape)
trainer = CNetTrainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=10, tag='inv_tv_sort',
                      lr_policy='linear', optimizer='adam', init_lr=0.0003, end_lr=0.00003)
# ckpt = os.path.join(LOG_DIR, 'imagenet_SDNet_res1_default_baseline_finetune_conv_5/model.ckpt-324174')
ckpt = os.path.join(LOG_DIR, 'imagenet_AlexNet_sorted_alex_sorted_finetune_conv_4/model.ckpt-450360')

trainer.train_inverter(ckpt)
