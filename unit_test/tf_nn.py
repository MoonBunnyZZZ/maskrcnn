import tensorflow as tf

x = tf.keras.layers.Input(shape=[None, None, 3])
y = tf.keras.layers.Conv2D(3, 3)(x)

m = tf.keras.Model(inputs=[x], outputs=[y])
m.summary()
