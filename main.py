import tensorflow as tf
from tensorflow import keras
from network.rpn import RPN
from network.backbone import FeaturePyramid


class MaskRCNN(keras.Model):
    def __init__(self):
        super(MaskRCNN, self).__init__()
        self.fpn = FeaturePyramid()
        self.rpn = RPN(256, num_anchors)

    def call(self, x):
        features = self.fpn(x, training=training)
