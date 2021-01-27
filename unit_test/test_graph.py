from tensorflow import keras
import tensorflow as tf


class Dummy(keras.layers.Layer):
    def __init__(self):
        super(Dummy, self).__init__()
        self.conv = keras.layers.Conv2D(10, 3)

    def call(self, inputs):
        output = list()
        for i in inputs:
            output.append(self.conv(i))
        return output


def loop_in_layer():
    inp1 = keras.layers.Input(shape=(448, 448, 3), name='image1')
    inp2 = keras.layers.Input(shape=(448, 448, 3), name='image2')
    inp3 = keras.layers.Input(shape=(448, 448, 3), name='image3')
    dummy = Dummy()
    out = dummy([inp1, inp2, inp3])
    m = keras.Model(inputs=[inp1, inp2, inp3], outputs=out)
    m.summary(line_length=150)


def log2_graph(x):
    """Implementation of Log2. TF doesn't have a native implementation."""
    return tf.log(x) / tf.log(2.0)


def net(boxes, feature_maps):
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
    h = y2 - y1
    w = x2 - x1
    # Use shape of first image. Images in a batch must have the same size.

    # Equation 1 in the Feature Pyramid Networks paper. Account for
    # the fact that our coordinates are normalized here.
    # e.g. a 224x224 ROI (in pixels) maps to P4
    image_area = tf.constant(123.)
    roi_level = tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area))
    roi_level = tf.minimum(5, tf.maximum(2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
    roi_level = tf.squeeze(roi_level, 2)  # [batch, num_boxes]

    # Loop through levels and apply ROI pooling to each. P2 to P5.
    pooled = []
    box_to_level = []
    for i, level in enumerate(range(2, 4)):
        ix = tf.where(tf.equal(roi_level, level), name='ix')
        level_boxes = tf.gather_nd(boxes, ix, name='get_level_box')

        # Box indices for crop_and_resize.
        box_indices = tf.cast(ix[:, 0], tf.int32, name='box_indices')
        return [box_indices, level_boxes]
        # Keep track of which box is mapped to which level
        box_to_level.append(ix)

        # Stop gradient propogation to ROI proposals
        level_boxes = tf.stop_gradient(level_boxes)
        box_indices = tf.stop_gradient(box_indices)

        # Crop and Resize
        # From Mask R-CNN paper: "We sample four regular locations, so
        # that we can evaluate either max or average pooling. In fact,
        # interpolating only a single value at each bin center (without
        # pooling) is nearly as effective."
        #
        # Here we use the simplified approach of a single value per bin,
        # which is how it's done in tf.crop_and_resize()
        # Result: [batch * num_boxes, pool_height, pool_width, channels]
        pooled.append(tf.image.crop_and_resize(
            feature_maps[i], level_boxes, box_indices, (14, 14),
            method="bilinear"))

    # Pack pooled features into one tensor
    pooled = tf.concat(pooled, axis=0, name='pooled')

    # Pack box_to_level mapping into one array and add another
    # column representing the order of pooled boxes
    box_to_level = tf.concat(box_to_level, axis=0)
    box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
    box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range], axis=1)

    # Rearrange pooled features to match the order of the original boxes
    # Sort box_to_level by batch then box index
    # TF doesn't have a way to sort by two columns, so merge them and sort.
    sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
    ix = tf.nn.top_k(sorting_tensor, k=tf.shape(
        box_to_level)[0]).indices[::-1]
    ix = tf.gather(box_to_level[:, 2], ix)
    pooled = tf.gather(pooled, ix)

    # Re-add the batch dimension
    shape = tf.concat([tf.shape(boxes)[:2], tf.shape(pooled)[1:]], axis=0)
    pooled = tf.reshape(pooled, shape)
    return pooled


box = keras.layers.Input(shape=(448, 4), name='image1')
feature1 = keras.layers.Input(shape=(448, 448, 3), name='image2')
feature2 = keras.layers.Input(shape=(448, 448, 3), name='image3')
feature = [feature1, feature2]

out = net(box, feature)
m = keras.Model(inputs=[box, feature1, feature2], outputs=out)
m.summary(line_length=150)
