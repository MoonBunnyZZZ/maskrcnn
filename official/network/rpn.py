import tensorflow as tf
from tensorflow.keras.layers import Conv2D

from utils import box_utils
from core import nms


def rpn(x, in_channels, num_anchors):
    p3, p4, p5, p6 = x
    conv = Conv2D(in_channels, 3, 1, 'same', activation='relu', name='rpn_conv')
    cls_conv = Conv2D(num_anchors, 1, 1, 'same', activation='sigmoid', name='rpn_cls_conv')
    reg_conv = Conv2D(num_anchors * 4, 1, 1, 'same', name='rpn_reg_conv')

    cls_out = dict()
    reg_out = dict()
    for i, feature in enumerate([p3, p4, p5, p6]):
        m = conv(feature)
        y_cls = cls_conv(m)
        y_reg = reg_conv(m)
        cls_out[i + 3] = y_cls
        reg_out[i + 3] = y_reg

    return [cls_out, reg_out]


def multilevel_propose_rois(rpn_boxes, rpn_scores, anchor_boxes, image_shape,
                            rpn_pre_nms_top_k=2000,
                            rpn_post_nms_top_k=1000,
                            rpn_nms_threshold=0.7,
                            rpn_score_threshold=0.0,
                            rpn_min_size_threshold=0.0,
                            decode_boxes=True,
                            clip_boxes=True,
                            use_batched_nms=False,
                            apply_sigmoid_to_score=True):
    rois = []
    roi_scores = []
    image_shape = tf.expand_dims(image_shape, axis=1)
    for level in sorted(rpn_scores.keys()):
        _, feature_h, feature_w, num_anchors_per_location = (
            rpn_scores[level].get_shape().as_list())
        num_boxes = feature_h * feature_w * num_anchors_per_location
        this_level_scores = tf.reshape(rpn_scores[level], [-1, num_boxes])
        this_level_boxes = tf.reshape(rpn_boxes[level], [-1, num_boxes, 4])
        this_level_anchors = tf.cast(
            tf.reshape(anchor_boxes[level], [-1, num_boxes, 4]),
            dtype=this_level_scores.dtype)

        if apply_sigmoid_to_score:
            this_level_scores = tf.sigmoid(this_level_scores)

        if decode_boxes:
            this_level_boxes = box_utils.decode_boxes(this_level_boxes,
                                                      this_level_anchors)
        if clip_boxes:
            this_level_boxes = box_utils.clip_boxes(this_level_boxes, image_shape)

        if rpn_min_size_threshold > 0.0:
            this_level_boxes, this_level_scores = box_utils.filter_boxes(
                this_level_boxes, this_level_scores, image_shape,
                rpn_min_size_threshold)

        this_level_pre_nms_top_k = min(num_boxes, rpn_pre_nms_top_k)
        this_level_post_nms_top_k = min(num_boxes, rpn_post_nms_top_k)
        if rpn_nms_threshold > 0.0:
            if use_batched_nms:
                this_level_rois, this_level_roi_scores, _, _ = (
                    tf.image.combined_non_max_suppression(
                        tf.expand_dims(this_level_boxes, axis=2),
                        tf.expand_dims(this_level_scores, axis=-1),
                        max_output_size_per_class=this_level_pre_nms_top_k,
                        max_total_size=this_level_post_nms_top_k,
                        iou_threshold=rpn_nms_threshold,
                        score_threshold=rpn_score_threshold,
                        pad_per_class=False,
                        clip_boxes=False))
            else:
                if rpn_score_threshold > 0.0:
                    this_level_boxes, this_level_scores = (
                        box_utils.filter_boxes_by_scores(this_level_boxes,
                                                         this_level_scores,
                                                         rpn_score_threshold))
                this_level_boxes, this_level_scores = box_utils.top_k_boxes(
                    this_level_boxes, this_level_scores, k=this_level_pre_nms_top_k)
                this_level_roi_scores, this_level_rois = (
                    nms.sorted_non_max_suppression_padded(
                        this_level_scores,
                        this_level_boxes,
                        max_output_size=this_level_post_nms_top_k,
                        iou_threshold=rpn_nms_threshold))
        else:
            this_level_rois, this_level_roi_scores = box_utils.top_k_boxes(
                this_level_rois, this_level_scores, k=this_level_post_nms_top_k)

        rois.append(this_level_rois)
        roi_scores.append(this_level_roi_scores)

    all_rois = tf.concat(rois, axis=1)
    all_roi_scores = tf.concat(roi_scores, axis=1)

    with tf.name_scope('top_k_rois'):
        _, num_valid_rois = all_roi_scores.get_shape().as_list()
        overall_top_k = min(num_valid_rois, rpn_post_nms_top_k)

        selected_rois, selected_roi_scores = box_utils.top_k_boxes(
            all_rois, all_roi_scores, k=overall_top_k)

    return selected_rois, selected_roi_scores
