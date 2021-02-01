import tensorflow as tf


def shared_neck(feature, fc_dims=1024):
    batch_size, num_proposal, h, w, c = feature.get_shape().as_list()
    feature = tf.reshape(feature, [batch_size, num_proposal, h * w * c])
    feature = tf.keras.layers.Dense(fc_dims)(feature)
    feature = tf.keras.layers.Dense(fc_dims)(feature)

    return feature


def predictor(feature, fc_dims=1024):
    cls = tf.keras.layers.Dense(fc_dims)(feature)
    reg = tf.keras.layers.Dense(fc_dims * 4)(feature)

    return cls, reg
