import tensorflow as tf
from tensorflow import keras
from network.rpn import RPN
from network.backbone import resnet, fpn


def maskrcnn():
    inputs = keras.layers.Input(shape=(448, 448, 3))
    y3, y4, y5 = resnet(inputs)
    p3, p4, p5, p6 = fpn(y3, y4, y5)
    # cls, reg = rpn([p3, p4, p5, p6])
    rpn = RPN(256, 3)
    cls_reg = rpn([p3, p4, p5, p6])
    model = keras.Model(inputs=inputs, outputs=cls_reg)
    # model.summary()
    return model


m = maskrcnn()
for i in range(len(m.layers)):
    print(m.get_layer(index=i).output)
# keras.utils.plot_model(m, to_file='model.png')


# m.compile()
keras.layers.Lambda
