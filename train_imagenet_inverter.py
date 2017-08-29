from Preprocessor import Preprocessor
from train.AlexNetInverter_Trainer import CNetTrainer
from datasets.ImageNet import ImageNet
from models.AlexNetInverter import AlexNetInverter

target_shape = [128, 128, 3]
model = AlexNetInverter(batch_size=32, target_shape=target_shape, num_layers=5, layer_id=1)
data = ImageNet()
preprocessor = Preprocessor(target_shape=target_shape)
trainer = CNetTrainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=10, tag='inv_tv',
                      lr_policy='linear', optimizer='adam', init_lr=0.0003, end_lr=0.00003)
ckpt = '/Data/Logs/CNet/imagenet_SDNet_res1_default_baseline_finetune_conv_5/model.ckpt-324174'

trainer.train_inverter(ckpt)
