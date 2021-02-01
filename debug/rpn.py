from tensorflow.keras.layers import Conv2D
import tensorflow as tf

from debug.box_ops import clip_box, reg_to_box, top_k_boxes


def rpn(x, in_channels, num_anchors):
    p3, p4, p5, p6 = x
    conv = Conv2D(in_channels, 3, 1, 'same', activation='relu', name='rpn_conv')
    cls_conv = Conv2D(num_anchors, 1, 1, 'same', activation='sigmoid', name='rpn_cls_conv')
    reg_conv = Conv2D(num_anchors * 4, 1, 1, 'same', name='rpn_reg_conv')

    cls_out = list()
    reg_out = list()
    for feature in [p3, p4, p5, p6]:
        m = conv(feature)
        y_cls = cls_conv(m)
        y_reg = reg_conv(m)
        cls_out.append(y_cls)
        reg_out.append(y_reg)

    return cls_out, reg_out


def generate_proposal(cls, reg, anchor, pre_nms_top_k, post_nms_top_k, image_size, iou_thresh, length_thresh):
    """

    :param cls: list, len(cls) is fpn level num, each in cls is tensor of shape [batch_size, num_anchor_this_level, 1]
    :param reg: list, len(cls) is fpn level num, each in reg is tensor of shape [batch_size, num_anchor_this_level, 4]
    :param anchor: list, len(anchor) is fpn level num, each in cls is tensor of shape [num_anchor_this_level, 4]
    :param pre_nms_top_k: int
    :param post_nms_top_k: int
    :param image_size: tuple (mage_h, image_w)
    :param length_thresh: float length threshold for filtering tiny box
    :param iou_thresh: float
    :return:
    """
    image_h, image_w = image_size
    boxes, scores = list(), list()
    anchor = tf.split(anchor, [3 * 56 * 56, 3 * 28 * 28, 3 * 14 * 14, 3 * 7 * 7], axis=0)
    for i, cls_level in enumerate(cls):
        cls_level = tf.sigmoid(cls_level)
        batch_size, feature_h, feature_w, num_anchor = cls_level.get_shape().as_list()
        cls_level = tf.reshape(cls_level, [batch_size, feature_h * feature_h * num_anchor, 1])
        reg_level = tf.reshape(reg[i], [batch_size, feature_h * feature_h * num_anchor, 4])

        anchor_level = anchor[i]
        num_boxes = anchor_level.get_shape().as_list()[0]

        box_level = reg_to_box(reg_level, anchor_level, batch_size)
        box_level = clip_box(box_level, image_h, image_w, 'level{}'.format(i))
        # box_level = remove_tiny(box_level, cls_level, length_thresh)

        level_pre_nms_top_k = min(num_boxes, pre_nms_top_k)
        level_post_nms_top_k = min(num_boxes, post_nms_top_k)
        level_box, level_score, _, _ = (
            tf.image.combined_non_max_suppression(tf.expand_dims(box_level, axis=2),
                                                  # tf.expand_dims(cls_level, axis=-1),
                                                  cls_level,
                                                  max_output_size_per_class=level_pre_nms_top_k,
                                                  max_total_size=level_post_nms_top_k,
                                                  iou_threshold=iou_thresh,
                                                  score_threshold=0.0,
                                                  clip_boxes=False))
        boxes.append(level_box)
        scores.append(level_score)

    boxes = tf.concat(boxes, axis=1)
    scores = tf.concat(scores, axis=1)

    num_valid_boxes = boxes.get_shape().as_list()[-2]
    overall_top_k = min(num_valid_boxes, post_nms_top_k)

    selected_boxes, selected_scores = top_k_boxes(boxes, scores, k=overall_top_k)
    return selected_boxes, selected_scores
