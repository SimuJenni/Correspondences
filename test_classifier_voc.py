from Preprocessor import Preprocessor
from eval.CNetTester import SDNetTester
from datasets.VOC2007 import VOC2007
from models.AlexNet_chan_rand_32 import AlexNet

target_shape = [227, 227, 3]
model = AlexNet(batch_size=1)
data = VOC2007()
preprocessor = Preprocessor(target_shape=target_shape)
tester = SDNetTester(model, data, preprocessor, tag='alex_random')

tester.test_classifier_voc(num_conv_trained=5)
