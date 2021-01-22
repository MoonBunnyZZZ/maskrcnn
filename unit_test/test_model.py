import tensorflow as tf
from tensorflow import keras


class Dummy(keras.Model):
    def __init__(self):
        super(Dummy, self).__init__()
        self.conv_c3_1x1 = keras.layers.Conv2D(1, 1, 1, "same")
        self.conv_c4_1x1 = keras.layers.Conv2D(1, 1, 1, "same")
        self.conv_c5_1x1 = keras.layers.Conv2D(1, 1, 1, "same")
        self.conv_c3_3x3 = keras.layers.Conv2D(1, 3, 1, "same")
        self.conv_c4_3x3 = keras.layers.Conv2D(1, 3, 1, "same")
        self.conv_c5_3x3 = keras.layers.Conv2D(1, 3, 1, "same")
        self.pool_c5_1x1 = keras.layers.MaxPool2D((1, 1), 2)
        self.upsample_2x = keras.layers.UpSampling2D(2)

    def call(self, c3_output, c4_output, c5_output):
        p3_output = self.conv_c3_1x1(c3_output)
        p4_output = self.conv_c4_1x1(c4_output)
        p5_output = self.conv_c5_1x1(c5_output)
        # p4_output = p4_output + self.upsample_2x(p5_output)
        # p3_output = p3_output + self.upsample_2x(p4_output)
        # p3_output = self.conv_c3_3x3(p3_output)
        # p4_output = self.conv_c4_3x3(p4_output)
        # p5_output = self.conv_c5_3x3(p5_output)
        # p6_output = self.pool_c5_1x1(c5_output)
        # return p3_output, p4_output, p5_output, p6_output

        return p3_output, p4_output, p5_output


# dummy1 = tf.ones((1, 3, 3, 3))
# dummy2 = tf.ones((1, 3, 3, 3))
# dummy3 = tf.ones((1, 3, 3, 3))
# model = Dummy()
# output = model(dummy1, dummy2, dummy3)
# # print(output[0])
# # print(len(output))
# model.compile()
# print(model.output)

backbone = keras.applications.ResNet50(include_top=False, input_shape=[224, 224, 3], weights=None)
backbone.summary()