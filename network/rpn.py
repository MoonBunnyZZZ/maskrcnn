from tensorflow import keras
from tensorflow.keras.layers import Layer, Conv2D, ReLU
import tensorflow as tf


class RPN(Layer):
    def __init__(self, in_channels, num_anchors):
        super(RPN, self).__init__()
        self.conv = Conv2D(in_channels, 3, 1, 'same', activation='relu', name='rpn_conv')
        self.cls = Conv2D(num_anchors, 1, 1, 'same', name='rpn_cls_conv')
        self.reg = Conv2D(num_anchors * 4, 1, 1, 'same', name='rpn_reg_conv')

    def call(self, x):
        logits = []
        bbox_reg = []
        for feature in x:
            t = self.conv(feature)
            logits.append(self.cls(t))
            bbox_reg.append(self.reg(t))
        return logits+bbox_reg


# if __name__ == '__main__':
#     rpn = RPN(32, 3)
#     print(rpn.trainable)
