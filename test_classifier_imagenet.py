from Preprocessor import Preprocessor
from eval.CNetTester import SDNetTester
from datasets.ImageNet import ImageNet
from models.AlexNet_chan_sort_2_bn1 import AlexNet

target_shape = [224, 224, 3]
model = AlexNet(batch_size=128)
data = ImageNet()
preprocessor = Preprocessor(target_shape=target_shape)
tester = SDNetTester(model, data, preprocessor, tag='alex_sorted')

ckpt = '/Data/Logs/CNet/imagenet_SDNet_res1_default_baseline_finetune_conv_5/'

tester.test_classifier(num_conv_trained=5)
