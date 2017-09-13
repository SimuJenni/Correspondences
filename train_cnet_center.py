from Preprocessor_tiles_puzzle import Preprocessor
from train.CNetTrainer_random import CNetTrainer
from datasets.ImageNet import ImageNet
from constants import LOG_DIR
from models.CNet_center_classifier import CNet
import os


target_shape = [72, 72, 3]
model = CNet(batch_size=128, target_shape=target_shape)
data = ImageNet()
preprocessor = Preprocessor(target_shape=target_shape, crop_size=[255, 255], augment_color=False)
trainer = CNetTrainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=180, tag='center',
                      lr_policy='linear', optimizer='adam', init_lr=0.0003, end_lr=0.000003)
trainer.train()
