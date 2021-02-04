import collections
import tensorflow as tf


class Anchor(object):
    def __init__(self, min_level, max_level, num_scales, aspect_ratios,
                 anchor_size, image_size):
        """Constructs multiscale anchors.

        Args:
          min_level: integer number of minimum level of the output feature pyramid.
          max_level: integer number of maximum level of the output feature pyramid.
          num_scales: integer number representing intermediate scales added on each
            level. For instances, num_scales=2 adds one additional intermediate
            anchor scales [2^0, 2^0.5] on each level.
          aspect_ratios: list of float numbers representing the aspect ratio anchors
            added on each level. The number indicates the ratio of width to height.
            For instances, aspect_ratios=[1.0, 2.0, 0.5] adds three anchors on each
            scale level.
          anchor_size: float number representing the scale of size of the base
            anchor to the feature stride 2^level.
          image_size: a list of integer numbers or Tensors representing [height,
            width] of the input image size.The image_size should be divisible by the
            largest feature stride 2^max_level.
        """
        self.min_level = min_level
        self.max_level = max_level
        self.num_scales = num_scales
        self.aspect_ratios = aspect_ratios
        self.anchor_size = anchor_size
        self.image_size = image_size
        self.boxes = self._generate_boxes()

    def _generate_boxes(self):
        """Generates multiscale anchor boxes.

        Returns:
          a Tensor of shape [N, 4], represneting anchor boxes of all levels
          concatenated together.
        """
        boxes_all = []
        for level in range(self.min_level, self.max_level + 1):
            boxes_l = []
            for scale in range(self.num_scales):
                for aspect_ratio in self.aspect_ratios:
                    stride = 2 ** level
                    intermediate_scale = 2 ** (scale / float(self.num_scales))
                    base_anchor_size = self.anchor_size * stride * intermediate_scale
                    aspect_x = aspect_ratio ** 0.5
                    aspect_y = aspect_ratio ** -0.5
                    half_anchor_size_x = base_anchor_size * aspect_x / 2.0
                    half_anchor_size_y = base_anchor_size * aspect_y / 2.0
                    x = tf.range(stride / 2, self.image_size[1], stride)
                    y = tf.range(stride / 2, self.image_size[0], stride)
                    xv, yv = tf.meshgrid(x, y)
                    xv = tf.cast(tf.reshape(xv, [-1]), dtype=tf.float32)
                    yv = tf.cast(tf.reshape(yv, [-1]), dtype=tf.float32)
                    # Tensor shape Nx4.
                    boxes = tf.stack([
                        yv - half_anchor_size_y, xv - half_anchor_size_x,
                        yv + half_anchor_size_y, xv + half_anchor_size_x
                    ],
                        axis=1)
                    boxes_l.append(boxes)
            # Concat anchors on the same level to tensor shape NxAx4.
            boxes_l = tf.stack(boxes_l, axis=1)
            boxes_l = tf.reshape(boxes_l, [-1, 4])
            boxes_all.append(boxes_l)
        return tf.concat(boxes_all, axis=0)

    def unpack_labels(self, labels):
        """Unpacks an array of labels into multiscales labels."""
        unpacked_labels = collections.OrderedDict()
        count = 0
        for level in range(self.min_level, self.max_level + 1):
            feat_size_y = tf.cast(self.image_size[0] / 2 ** level, tf.int32)
            feat_size_x = tf.cast(self.image_size[1] / 2 ** level, tf.int32)
            steps = feat_size_y * feat_size_x * self.anchors_per_location
            unpacked_labels[level] = tf.reshape(labels[count:count + steps],
                                                [feat_size_y, feat_size_x, -1])
            count += steps
        return unpacked_labels

    @property
    def anchors_per_location(self):
        return self.num_scales * len(self.aspect_ratios)

    @property
    def multilevel_boxes(self):
        return self.unpack_labels(self.boxes)
