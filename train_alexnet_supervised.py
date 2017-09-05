from Preprocessor import Preprocessor
from datasets.ImageNet import ImageNet
from eval.AlexTrainer_sorted import CNetTrainer
from models.AlexNet_layers import AlexNet
from constants import LOG_DIR
import os

target_shape = [224, 224, 3]
model = AlexNet(batch_size=256)
data = ImageNet()
preprocessor = Preprocessor(target_shape=target_shape, augment_color=False)
trainer = CNetTrainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=90, tag='alex_supervised',
                      lr_policy='linear', optimizer='adam', init_lr=0.001, end_lr=0.00001)
chpt_path = os.path.join(LOG_DIR, 'imagenet_SDNet_res1_default_baseline_finetune_conv_5/model.ckpt-324174')
trainer.transfer_finetune(chpt_path, num_conv2train=5, num_conv2init=0)
