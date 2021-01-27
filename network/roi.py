import tensorflow as tf

# tf.image.crop_and_resize
x = tf.constant([[[[1.], [2.]], [[3], [4]]]])
box = tf.constant([[0.2, 0.3, 0.4, 0.5]])
box_ind = tf.constant([0])
i = 4
crop_size = tf.constant([i, i])

res = tf.image.crop_and_resize(x, box, box_indices=box_ind, crop_size=crop_size)
print(res)


def roi_align_pool(proposal, feature, level):
    # proposal [None, 2000, 4]
    # feature [p3, p4, p5, p6], p [None, H, W, C]

    pass


class ROIAlign(tf.keras.layers.Layer):
    def __init__(self):
        super(ROIAlign, self).__init__()

    def call(self, inputs):
        feature, proposal = inputs

        p3, p4, p5, p6 = feature
