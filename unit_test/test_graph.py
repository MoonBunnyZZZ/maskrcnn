from tensorflow import keras

from network.backbone import FeaturePyramid
from network.rpn import RPN


class MaskRCNN(keras.Model):
    def __init__(self):
        super(MaskRCNN, self).__init__()
        self.fpn = FeaturePyramid()
        self.rpn = RPN(256, 9)

    def call(self, x):
        features = self.fpn(x, training=True)
        return features


m = MaskRCNN()
