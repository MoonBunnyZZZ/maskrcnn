import tensorflow as tf
from tensorflow import keras


def block(x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
    bn_axis = 3

    if conv_shortcut is True:
        shortcut = keras.layers.Conv2D(4 * filters, 1, strides=stride, name=name + '_0_conv')(x)
        shortcut = keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(shortcut)
    else:
        shortcut = x

    x = keras.layers.Conv2D(filters, 1, strides=stride, name=name + '_1_conv')(x)
    x = keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
    x = keras.layers.Activation('relu', name=name + '_1_relu')(x)

    x = keras.layers.Conv2D(filters, kernel_size, padding='SAME', name=name + '_2_conv')(x)
    x = keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
    x = keras.layers.Activation('relu', name=name + '_2_relu')(x)

    x = keras.layers.Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
    x = keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_3_bn')(x)

    x = keras.layers.Add(name=name + '_add')([shortcut, x])
    x = keras.layers.Activation('relu', name=name + '_out')(x)
    return x


def stack(x, filters, blocks, stride1=2, name=None):
    x = block(x, filters, stride=stride1, name=name + '_block1')
    for i in range(2, blocks + 1):
        x = block(x, filters, conv_shortcut=False, name=name + '_block' + str(i))
    return x


def stack_fn(x):
    x1 = stack(x, 64, 3, stride1=1, name='stack2')
    x2 = stack(x1, 128, 4, name='stack3')
    x3 = stack(x2, 256, 6, name='stack4')
    x4 = stack(x3, 512, 3, name='stack5')
    return x2, x3, x4


def resnet(img_input):
    x = keras.layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad')(img_input)
    x = keras.layers.Conv2D(64, 7, strides=2, name='conv1_conv')(x)

    x = keras.layers.BatchNormalization(axis=3, epsilon=1.001e-5, name='conv1_bn')(x)
    x = keras.layers.Activation('relu', name='conv1_relu')(x)

    x = keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
    x = keras.layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)

    c3, c4, c5 = stack_fn(x)

    return c3, c4, c5


def fpn(c3, c4, c5):
    p3_output = keras.layers.Conv2D(256, 1, 1, "same", name="c3_conv_1x1")(c3)
    p4_output = keras.layers.Conv2D(256, 1, 1, "same", name="c4_conv_1x1")(c4)
    p5_output = keras.layers.Conv2D(256, 1, 1, "same", name="c5_conv_1x1")(c5)
    p4_output = p4_output + keras.layers.UpSampling2D(2)(p5_output)
    p3_output = p3_output + keras.layers.UpSampling2D(2)(p4_output)
    p3_output = keras.layers.Conv2D(256, 3, 1, "same", name="p3_conv_3x3")(p3_output)
    p4_output = keras.layers.Conv2D(256, 3, 1, "same", name="p4_conv_3x3")(p4_output)
    p5_output = keras.layers.Conv2D(256, 3, 1, "same", name="p5_conv_3x3")(p5_output)
    p6_output = keras.layers.MaxPool2D((1, 1), 2, name="p6_maxpool")(c5)
    return p3_output, p4_output, p5_output, p6_output


if __name__ == '__main__':
    # inputs = keras.layers.Input(shape=(224, 224, 3))
    # y3, y4, y5 = resnet(inputs)
    # p3, p4, p5, p6 = fpn(y3, y4, y5)
    # model = keras.Model(inputs=inputs, outputs=[p3, p4, p5, p6])
    # model.summary()
    # print(model.outputs)
    # print(model.output)
    pass
