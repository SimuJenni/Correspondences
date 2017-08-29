from Preprocessor_tiles import Preprocessor
from train.CNetTrainer_class_sobel import CNetTrainer
from datasets.ImageNet import ImageNet
from models.CNet_class_sobel import CNet

target_shape = [64, 64, 3]
model = CNet(batch_size=64, target_shape=target_shape)
data = ImageNet()
preprocessor = Preprocessor(target_shape=target_shape, crop_size=[192, 192], augment_color=False)
trainer = CNetTrainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=90, tag='vgg_a',
                      lr_policy='linear', optimizer='adam', init_lr=0.001, end_lr=0.00001)
trainer.train()
