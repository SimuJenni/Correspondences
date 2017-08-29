from Preprocessor import Preprocessor
from datasets.ImageNet import ImageNet
from eval.AlexTrainer_sorted import CNetTrainer
from models.AlexNet_chan_sort import AlexNet

target_shape = [224, 224, 3]
model = AlexNet(batch_size=256)
data = ImageNet()
preprocessor = Preprocessor(target_shape=target_shape, augment_color=False)
trainer = CNetTrainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=90, tag='alex_sorted',
                      lr_policy='linear', optimizer='adam', init_lr=0.001, end_lr=0.00001)
chpt_path = '/Data/Logs/CNet/imagenet_SDNet_res1_default_baseline_finetune_conv_5/model.ckpt-324174'
trainer.transfer_finetune(chpt_path, num_conv2train=4, num_conv2init=1)
