import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.python.keras.utils import losses_utils


class RPNClsLoss(Layer):
    def __init__(self):
        super(RPNClsLoss, self).__init__(name="RPNClassLoss")
        self.fn = tf.keras.losses.BinaryCrossentropy(
            reduction=losses_utils.ReductionV2.AUTO)

    def call(self, inputs):
        y_true, y_pred, sample_idx = inputs
        y_true = y_true
        y_pred = y_pred

        cross_entropy = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        cross_entropy = tf.reduce_sum(cross_entropy, axis=-1, name='rpn_cls_sum')
        cross_entropy = tf.multiply(cross_entropy, sample_idx, name='rpn_cls_multi')
        # self.add_loss(cross_entropy, inputs=True)
        return tf.reduce_sum(cross_entropy, name='rpn_cls_sum_all')


class RPNRegLoss(Layer):
    def __init__(self, delta=0.5):
        super(RPNRegLoss, self).__init__(name="RPNRegressionLoss")
        self._delta = delta

    def call(self, inputs):
        y_true, y_pred, pos_sample_idx = inputs

        difference = y_true - y_pred
        absolute_difference = tf.abs(difference)
        squared_difference = difference ** 2
        loss = tf.where(
            tf.less(absolute_difference, self._delta),
            0.5 * squared_difference,
            absolute_difference - 0.5,
        )
        loss = tf.multiply(loss, pos_sample_idx, name='rpn_reg_mul')
        # self.add_loss(loss, inputs=True)
        return tf.reduce_sum(loss)


class RPNLoss(Layer):
    def __init__(self):
        super(RPNLoss, self).__init__(name="RPNLoss")
        self.cls_loss = RPNClsLoss()
        self.reg_loss = RPNRegLoss()

    def call(self, inputs):
        cls_gt, cls_pred, reg_gt, reg_pred, sample_idx, pos_sample_idx = inputs
        cls_loss = self.cls_loss([cls_gt, cls_pred, sample_idx])
        reg_loss = self.reg_loss([reg_gt, reg_pred, pos_sample_idx])
        # loss = cls_loss + reg_loss
        self.add_loss(cls_loss, inputs=True)
        self.add_loss(reg_loss, inputs=True)
        return cls_loss, reg_loss
