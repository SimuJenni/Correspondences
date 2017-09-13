from Preprocessor_tiles_gray import Preprocessor
from eval.CNetTester_random import CNetTester
from datasets.ImageNet import ImageNet
from constants import LOG_DIR
from models.CNet_random_multi2 import CNet
import os


target_shape = [72, 72, 1]
model = CNet(batch_size=128, target_shape=target_shape)
data = ImageNet()
preprocessor = Preprocessor(target_shape=target_shape, crop_size=[255, 255], augment_color=False)
tester = CNetTester(model=model, dataset=data, pre_processor=preprocessor, tag='multi_new')
tester.test()
