from tensorflow import keras
from tensorflow.keras.layers import Layer, Conv2D, ReLU


class RPN(Layer):
    def __init__(self, in_channels, num_anchors):
        super(RPN, self).__init__()
        self.conv = Conv2D(in_channels, 3, 1, 'same', activation='relu')
        self.cls = Conv2D(num_anchors, 1, 1, 'same')
        self.reg = Conv2D(num_anchors * 4, 1, 1, 'same')

    def call(self, x):
        # logits = []
        # bbox_reg = []
        # for feature in x:
        #     t = F.relu(self.conv(feature))
        #     logits.append(self.cls_logits(t))
        #     bbox_reg.append(self.bbox_pred(t))
        # return logits, bbox_reg
        t = self.conv(x)
        cls = self.cls(t)
        reg = self.reg(t)
        return cls, reg


if __name__ == '__main__':
    rpn = RPN(32, 3)
    print(rpn.trainable)
