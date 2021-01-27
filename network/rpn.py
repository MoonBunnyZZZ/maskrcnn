from tensorflow.keras.layers import Layer, Conv2D, ReLU
import tensorflow as tf

from utils.box_ops import nms, remove_tiny, clip_box, reg_to_box


class RPN(Layer):
    def __init__(self, in_channels, num_anchors):
        super(RPN, self).__init__()
        self.conv = Conv2D(in_channels, 3, 1, 'same', activation='relu', name='rpn_conv')
        self.cls = Conv2D(num_anchors, 1, 1, 'same', name='rpn_cls_conv')
        self.reg = Conv2D(num_anchors * 4, 1, 1, 'same', name='rpn_reg_conv')

    def call(self, x):
        cls, reg = list(), list()
        p3, p4, p5, p6 = x

        for feature in x:
            t = self.conv(feature)

            cls_output = self.cls(t)
            _, h, w, c = cls_output.shape
            cls.append(tf.reshape(cls, [-1, h * w, c]))

            reg_output = self.reg(t)
            _, h, w, c = reg_output.shape
            reg.append(tf.reshape(reg, [-1, h * w, c]))

        cls = tf.concat(cls, axis=0)
        reg = tf.concat(reg, axis=0)
        return [cls, reg]


def rpn(x, in_channels, num_anchors):
    p3, p4, p5, p6 = x
    conv = Conv2D(in_channels, 3, 1, 'same', activation='relu', name='rpn_conv')
    cls_conv = Conv2D(num_anchors, 1, 1, 'same', activation='sigmoid', name='rpn_cls_conv')
    reg_conv = Conv2D(num_anchors * 4, 1, 1, 'same', name='rpn_reg_conv')

    m3 = conv(p3)
    y3_cls = cls_conv(m3)
    y3_reg = reg_conv(m3)
    m4 = conv(p4)
    y4_cls = cls_conv(m4)
    y4_reg = reg_conv(m4)
    m5 = conv(p5)
    y5_cls = cls_conv(m5)
    y5_reg = reg_conv(m5)
    m6 = conv(p6)
    y6_cls = cls_conv(m6)
    y6_reg = reg_conv(m6)

    n, h, w, c = m3.shape
    # print(tf.shape(m3).numpy())
    print(n, h, w, c)
    y3_cls = tf.reshape(y3_cls, [-1, h * w * num_anchors, 1], name='y3_cls_reshape')
    y3_reg = tf.reshape(y3_reg, [-1, h * w * num_anchors, 4], name='y3_reg_reshape')

    n, h, w, c = m4.shape
    y4_cls = tf.reshape(y4_cls, [-1, h * w * num_anchors, 1], name='y4_cls_reshape')
    y4_reg = tf.reshape(y4_reg, [-1, h * w * num_anchors, 4], name='y4_reg_reshape')

    n, h, w, c = m5.shape
    y5_cls = tf.reshape(y5_cls, [-1, h * w * num_anchors, 1], name='y5_cls_reshape')
    y5_reg = tf.reshape(y5_reg, [-1, h * w * num_anchors, 4], name='y5_reg_reshape')

    n, h, w, c = m6.shape
    y6_cls = tf.reshape(y6_cls, [-1, h * w * num_anchors, 1], name='y6_cls_reshape')
    y6_reg = tf.reshape(y6_reg, [-1, h * w * num_anchors, 4], name='y6_reg_reshape')

    rpn_cls = tf.concat([y3_cls, y4_cls, y5_cls, y6_cls], axis=1, name='rpn_cls_concat')
    rpn_reg = tf.concat([y3_reg, y4_reg, y5_reg, y6_reg], axis=1, name='rpn_reg_concat')
    return [rpn_cls, rpn_reg]


def generate_proposal(cls, reg, anchor, top_n, anchor_num_per_level, image_size, length_thre, iou_thre):
    cls = tf.reshape(cls, [-1, sum(anchor_num_per_level)])
    box = reg_to_box(reg, anchor)
    l3_cls, l4_cls, l5_cls, l6_cls = tf.split(cls, anchor_num_per_level, axis=1)
    _, y3_top_n_idx = tf.nn.top_k(l3_cls, top_n, name='level3_top_n')
    _, y4_top_n_idx = tf.nn.top_k(l4_cls, top_n, name='level4_top_n')
    _, y5_top_n_idx = tf.nn.top_k(l5_cls, top_n, name='level5_top_n')
    # _, y6_top_n_idx = tf.nn.top_k(l6_cls, top_n, name='level6_top_n')

    # top n
    l3_box, l4_box, l5_box, l6_box = tf.split(box, anchor_num_per_level, axis=1)
    l3_proposal = tf.gather(l3_box, y3_top_n_idx, axis=1, batch_dims=-1)
    l4_proposal = tf.gather(l4_box, y4_top_n_idx, axis=1, batch_dims=-1)
    l5_proposal = tf.gather(l5_box, y5_top_n_idx, axis=1, batch_dims=-1)

    l3_cls = tf.gather(l3_cls, y3_top_n_idx, axis=1, batch_dims=-1)
    l4_cls = tf.gather(l4_cls, y4_top_n_idx, axis=1, batch_dims=-1)
    l5_cls = tf.gather(l5_cls, y5_top_n_idx, axis=1, batch_dims=-1)

    # clip proposal
    h, w = image_size
    l3_proposal = clip_box(l3_proposal, h, w, 'l3')
    l4_proposal = clip_box(l4_proposal, h, w, 'l4')
    l5_proposal = clip_box(l5_proposal, h, w, 'l5')
    l6_proposal = clip_box(l6_box, h, w, 'l6')

    # remove tiny proposal
    l3_proposal, l3_cls = remove_tiny(l3_proposal, l3_cls, length_thre)
    l4_proposal, l4_cls = remove_tiny(l4_proposal, l4_cls, length_thre)
    l5_proposal, l5_cls = remove_tiny(l5_proposal, l5_cls, length_thre)
    l6_proposal, l6_cls = remove_tiny(l6_proposal, l6_cls, length_thre)

    # nms
    proposal = tf.concat([l3_proposal, l4_proposal, l5_proposal, l6_proposal], axis=1)
    cls = tf.concat([l3_cls, l4_cls, l5_cls, l6_cls], axis=1)
    proposal = nms(proposal, cls, top_n, iou_thre)
    # proposal = tf.expand_dims(proposal, -1)
    # proposal = tf.image.crop_and_resize(proposal,
    #                                     tf.constant([[0.2, 0.3, 0.4, 0.5]]),
    #                                     tf.constant([0]),
    #                                     tf.constant([2, 2]))
    return proposal
