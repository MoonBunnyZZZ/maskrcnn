import tensorflow as tf
from tensorflow.keras.losses import Loss


class RPNClsLoss(tf.losses.Loss):
    def __init__(self):
        super(RPNClsLoss, self).__init__(reduction="none", name="RPNClassLoss")

    def call(self, y_true, y_pred):
        cross_entropy0 = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true[0], logits=y_pred[0])
        cross_entropy1 = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
        cross_entropy2 = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
        cross_entropy3 = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
        return tf.reduce_sum(cross_entropy, axis=-1)


class RPNRegLoss(tf.losses.Loss):
    def __init__(self, delta=0.5):
        super(RPNRegLoss, self).__init__(reduction="none", name="RPNRegressionLoss")
        self._delta = delta

    def call(self, y_true, y_pred):
        difference = y_true - y_pred
        absolute_difference = tf.abs(difference)
        squared_difference = difference ** 2
        loss = tf.where(
            tf.less(absolute_difference, self._delta),
            0.5 * squared_difference,
            absolute_difference - 0.5,
        )
        return tf.reduce_sum(loss, axis=-1)


class RPNLoss(tf.losses.Loss):
    def __init__(self):
        super(RPNLoss, self).__init__(reduction="none", name="RPNClassLoss")
        self.cls_loss = RPNClsLoss()
        self.reg_loss = RPNRegLoss()

    def call(self, y_true, y_pred):
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
        return tf.reduce_sum(cross_entropy, axis=-1)
