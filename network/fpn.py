import tensorflow as tf


def fpn(backbone_feature):
    c3, c4, c5 = backbone_feature
    p3_output = tf.keras.layers.Conv2D(256, 1, 1, "same", name="c3_conv_1x1")(c3)
    p4_output = tf.keras.layers.Conv2D(256, 1, 1, "same", name="c4_conv_1x1")(c4)
    p5_output = tf.keras.layers.Conv2D(256, 1, 1, "same", name="c5_conv_1x1")(c5)
    p4_output = p4_output + tf.keras.layers.UpSampling2D(2)(p5_output)
    p3_output = p3_output + tf.keras.layers.UpSampling2D(2)(p4_output)
    p3_output = tf.keras.layers.Conv2D(256, 3, 1, "same", name="p3_conv_3x3")(p3_output)
    p4_output = tf.keras.layers.Conv2D(256, 3, 1, "same", name="p4_conv_3x3")(p4_output)
    p5_output = tf.keras.layers.Conv2D(256, 3, 1, "same", name="p5_conv_3x3")(p5_output)
    p6_output = tf.keras.layers.MaxPool2D((1, 1), 2, name="p6_maxpool")(p5_output)
    return [p3_output, p4_output, p5_output, p6_output]
