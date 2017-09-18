from Preprocessor import Preprocessor
from datasets.VOC2007 import VOC2007
from eval.AlexTrainer_sorted import CNetTrainer
from models.AlexNet_chan_rand_32 import AlexNet

target_shape = [227, 227, 3]
model = AlexNet(batch_size=16)
data = VOC2007()
preprocessor = Preprocessor(target_shape=target_shape, augment_color=False)
trainer = CNetTrainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=250, tag='alex_random2',
                      lr_policy='linear', optimizer='adam', init_lr=0.001, end_lr=0.00001)
trainer.transfer_finetune(None, num_conv2train=5, num_conv2init=0)

