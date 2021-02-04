from absl import logging
import tensorflow as tf


class RpnScoreLoss(object):
    def __init__(self, rpn_sample_size_per_im):
        self.rpn_sample_size_per_im = rpn_sample_size_per_im
        self._binary_crossentropy = tf.keras.losses.BinaryCrossentropy(
            reduction=tf.keras.losses.Reduction.SUM, from_logits=True)

    def __call__(self, score_outputs, labels):
        nor = tf.cast(tf.shape(score_outputs)[0] * self.rpn_sample_size_per_im, dtype=tf.float32)
        loss = self._rpn_score_loss(score_outputs, labels, nor)

        return loss

    def _rpn_score_loss(self, score_outputs, score_targets, normalizer=1.0):
        # score_targets has three values:
        # (1) score_targets[i]=1, the anchor is a positive sample.
        # (2) score_targets[i]=0, negative.
        # (3) score_targets[i]=-1, the anchor is don't care (ignore).
        mask = tf.math.logical_or(tf.math.equal(score_targets, 1),
                                  tf.math.equal(score_targets, 0))

        score_targets = tf.math.maximum(score_targets, tf.zeros_like(score_targets))
        score_targets = tf.expand_dims(score_targets, axis=-1)
        score_outputs = tf.expand_dims(score_outputs, axis=-1)
        score_loss = self._binary_crossentropy(score_targets, score_outputs, sample_weight=mask)

        score_loss /= normalizer
        return score_loss


class RpnBoxLoss(object):
    def __init__(self):
        self._huber_loss = tf.keras.losses.Huber(delta=1 / 9, reduction=tf.keras.losses.Reduction.SUM)

    def __call__(self, box_outputs, labels):
        loss = self._rpn_box_loss(box_outputs, labels)
        return loss

    def _rpn_box_loss(self, box_outputs, box_targets, normalizer=1.0):
        mask = tf.cast(tf.not_equal(box_targets, 0.0), dtype=tf.float32)
        box_targets = tf.expand_dims(box_targets, axis=-1)
        box_outputs = tf.expand_dims(box_outputs, axis=-1)
        box_loss = self._huber_loss(box_targets, box_outputs, sample_weight=mask)
        # The loss is normalized by the sum of non-zero weights and additional
        # normalizer provided by the function caller. Using + 0.01 here to avoid
        # division by zero.
        box_loss /= normalizer * (tf.reduce_sum(mask) + 0.01)
        return box_loss


class HeadClassLoss(object):
    def __init__(self):
        self._categorical_crossentropy = tf.keras.losses.CategoricalCrossentropy(
            reduction=tf.keras.losses.Reduction.SUM, from_logits=True)

    def __call__(self, class_outputs, class_targets):
        """Computes the class loss (Fast-RCNN branch) of Mask-RCNN.
        This function implements the classification loss of the Fast-RCNN.
        The classification loss is softmax on all RoIs.
        Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/modeling/fast_rcnn_heads.py  # pylint: disable=line-too-long

        Args:
          class_outputs: a float tensor representing the class prediction for each box
            with a shape of [batch_size, num_boxes, num_classes].
          class_targets: a float tensor representing the class label for each box
            with a shape of [batch_size, num_boxes].

        Returns:
          a scalar tensor representing total class loss.
        """
        batch_size, num_boxes, num_classes = class_outputs.get_shape().as_list()
        class_targets = tf.cast(class_targets, dtype=tf.int32)
        class_targets_one_hot = tf.one_hot(class_targets, num_classes)
        return self._fast_rcnn_class_loss(class_outputs, class_targets_one_hot,
                                          normalizer=batch_size * num_boxes / 2.0)

    def _fast_rcnn_class_loss(self, class_outputs, class_targets_one_hot,
                              normalizer):
        """Computes classification loss."""
        class_loss = self._categorical_crossentropy(class_targets_one_hot, class_outputs)
        class_loss /= normalizer
        return class_loss


class HeadBoxLoss(object):
    def __init__(self):
        self._huber_loss = tf.keras.losses.Huber(delta=1.0, reduction=tf.keras.losses.Reduction.SUM)

    def __call__(self, box_outputs, class_targets, box_targets):
        """Computes the box loss (Fast-RCNN branch) of Mask-RCNN.

        This function implements the box regression loss of the Fast-RCNN. As the
        `box_outputs` produces `num_classes` boxes for each RoI, the reference model
        expands `box_targets` to match the shape of `box_outputs` and selects only
        the target that the RoI has a maximum overlap. (Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/roi_data/fast_rcnn.py)  # pylint: disable=line-too-long
        Instead, this function selects the `box_outputs` by the `class_targets` so
        that it doesn't expand `box_targets`.

        The box loss is smooth L1-loss on only positive samples of RoIs.
        Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/modeling/fast_rcnn_heads.py  # pylint: disable=line-too-long

        Args:
          box_outputs: a float tensor representing the box prediction for each box
            with a shape of [batch_size, num_boxes, num_classes * 4].
          class_targets: a float tensor representing the class label for each box
            with a shape of [batch_size, num_boxes].
          box_targets: a float tensor representing the box label for each box
            with a shape of [batch_size, num_boxes, 4].

        Returns:
          box_loss: a scalar tensor representing total box regression loss.
        """
        class_targets = tf.cast(class_targets, dtype=tf.int32)

        # Selects the box from `box_outputs` based on `class_targets`, with which
        # the box has the maximum overlap.
        (batch_size, num_rois,
         num_class_specific_boxes) = box_outputs.get_shape().as_list()
        num_classes = num_class_specific_boxes // 4
        box_outputs = tf.reshape(box_outputs,
                                 [batch_size, num_rois, num_classes, 4])

        box_indices = tf.reshape(
            class_targets + tf.tile(
                tf.expand_dims(
                    tf.range(batch_size) * num_rois * num_classes, 1),
                [1, num_rois]) + tf.tile(
                tf.expand_dims(tf.range(num_rois) * num_classes, 0),
                [batch_size, 1]), [-1])

        box_outputs = tf.matmul(
            tf.one_hot(
                box_indices,
                batch_size * num_rois * num_classes,
                dtype=box_outputs.dtype), tf.reshape(box_outputs, [-1, 4]))
        box_outputs = tf.reshape(box_outputs, [batch_size, -1, 4])

        return self._fast_rcnn_box_loss(box_outputs, box_targets, class_targets)

    def _fast_rcnn_box_loss(self, box_outputs, box_targets, class_targets,
                            normalizer=1.0):
        """Computes box regression loss."""
        mask = tf.tile(tf.expand_dims(tf.greater(class_targets, 0), axis=2),
                       [1, 1, 4])
        mask = tf.cast(mask, dtype=tf.float32)
        box_targets = tf.expand_dims(box_targets, axis=-1)
        box_outputs = tf.expand_dims(box_outputs, axis=-1)
        box_loss = self._huber_loss(box_targets, box_outputs, sample_weight=mask)
        # The loss is normalized by the number of ones in mask,
        # additianal normalizer provided by the user and using 0.01 here to avoid
        # division by 0.
        box_loss /= normalizer * (tf.reduce_sum(mask) + 0.01)
        return box_loss


class MaskrcnnLoss(object):
    """Mask R-CNN instance segmentation mask loss function."""

    def __init__(self):
        self._binary_crossentropy = tf.keras.losses.BinaryCrossentropy(
            reduction=tf.keras.losses.Reduction.SUM, from_logits=True)

    def __call__(self, mask_outputs, mask_targets, select_class_targets):
        """Computes the mask loss of Mask-RCNN.

        This function implements the mask loss of Mask-RCNN. As the `mask_outputs`
        produces `num_classes` masks for each RoI, the reference model expands
        `mask_targets` to match the shape of `mask_outputs` and selects only the
        target that the RoI has a maximum overlap. (Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/roi_data/mask_rcnn.py)  # pylint: disable=line-too-long
        Instead, this implementation selects the `mask_outputs` by the `class_targets`
        so that it doesn't expand `mask_targets`. Note that the selection logic is
        done in the post-processing of mask_rcnn_fn in mask_rcnn_architecture.py.

        Args:
          mask_outputs: a float tensor representing the prediction for each mask,
            with a shape of
            [batch_size, num_masks, mask_height, mask_width].
          mask_targets: a float tensor representing the binary mask of ground truth
            labels for each mask with a shape of
            [batch_size, num_masks, mask_height, mask_width].
          select_class_targets: a tensor with a shape of [batch_size, num_masks],
            representing the foreground mask targets.

        Returns:
          mask_loss: a float tensor representing total mask loss.
        """
        with tf.name_scope('mask_rcnn_loss'):
            (batch_size, num_masks, mask_height,
             mask_width) = mask_outputs.get_shape().as_list()

            weights = tf.tile(
                tf.reshape(tf.greater(select_class_targets, 0),
                           [batch_size, num_masks, 1, 1]),
                [1, 1, mask_height, mask_width])
            weights = tf.cast(weights, dtype=tf.float32)

            mask_targets = tf.expand_dims(mask_targets, axis=-1)
            mask_outputs = tf.expand_dims(mask_outputs, axis=-1)
            mask_loss = self._binary_crossentropy(mask_targets, mask_outputs,
                                                  sample_weight=weights)

            # The loss is normalized by the number of 1's in weights and
            # + 0.01 is used to avoid division by zero.
            return mask_loss / (tf.reduce_sum(weights) + 0.01)
