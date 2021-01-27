"""
using in roi head for select train sample
"""
import numpy as np
import tensorflow as tf


def box_iou(boxes1, boxes2):
    b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 2),
                            [1, 1, 1, tf.shape(boxes2)[1]]),
                    [-1, tf.shape(boxes1)[1] * tf.shape(boxes2)[1], 4])
    b2 = tf.tile(boxes2, [1, tf.shape(boxes1)[1], 1])

    b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=2)
    b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=2)
    y1 = tf.maximum(b1_y1, b2_y1)
    x1 = tf.maximum(b1_x1, b2_x1)
    y2 = tf.minimum(b1_y2, b2_y2)
    x2 = tf.minimum(b1_x2, b2_x2)
    intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)

    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area + b2_area - intersection

    iou = intersection / union
    overlaps = tf.reshape(iou, [-1, tf.shape(boxes1)[1], tf.shape(boxes2)[1]])
    return overlaps


def match_proposal(match_quality_matrix, low_threshold=0.3, high_threshold=0.7):
    """
        match_quality_matrix shape:[batch_size, gt_num, 4]
    """
    matched_vals = tf.reduce_max(match_quality_matrix, axis=1)
    matches = tf.argmax(match_quality_matrix, axis=1)

    below_low_threshold = matched_vals < low_threshold
    between_thresholds = (matched_vals >= low_threshold) & (matched_vals < high_threshold)

    below_idx = tf.where(below_low_threshold)
    below_update = tf.tile(tf.constant(np.array([-1], dtype=np.int64)),
                           [tf.shape(below_idx)[0]])
    matches = tf.tensor_scatter_nd_update(matches, below_idx, below_update)

    between_idx = tf.where(between_thresholds)
    between_update = tf.tile(tf.constant(np.array([-2], dtype=np.int64)),
                             [tf.shape(between_idx)[0]])
    matches = tf.tensor_scatter_nd_update(matches, between_idx, between_update)

    return matches


def encode_flag_and_match_box(gt_box, proposal, gt_label):
    """ encode grid anchors with 1,0,-1 and match gt box to each proposal
        gt_box shape:[batch_size, gt_num, 4]
        gt_label shape:[batch_size, gt_num, 1]
        proposal shape:[batch_size, proposal_num, 4]
    """

    if tf.size(gt_box) == 0:
        clipped_matched_idxs = tf.constant(np.zeros(tf.shape(proposal)))
        flags = tf.constant(np.zeros((tf.shape(proposal)[0],)))
    else:
        match_quality_matrix = box_iou(gt_box, proposal)
        matched_idxs = match_proposal(match_quality_matrix)  # shape:[batch_size, proposal_num]

        # clipped_matched_idxs shape [batch_size, proposal_num]
        clipped_matched_idxs = tf.clip_by_value(matched_idxs, 0, 1000)
        flags = tf.gather(gt_label, clipped_matched_idxs, batch_dims=1)
        flags = tf.squeeze(flags, -1)
        flags = tf.cast(flags, tf.int64)

        bg_idx = tf.where(tf.equal(matched_idxs, -1))
        bg_update = tf.tile(tf.constant(np.array([0], dtype=np.int64)),
                            [tf.shape(bg_idx)[0]])
        flags = tf.tensor_scatter_nd_update(flags, bg_idx, bg_update)

        idx_to_discard = tf.where(tf.equal(matched_idxs, -2))
        discard_update = tf.tile(tf.constant(np.array([-1], dtype=np.int64)),
                                 [tf.shape(idx_to_discard)[0]])
        flags = tf.tensor_scatter_nd_update(flags, idx_to_discard, discard_update)

    return clipped_matched_idxs, flags


# boxes1 = tf.constant(np.random.rand(2, 3, 4))
# boxes2 = tf.constant(np.random.rand(2, 3, 4))
# boxes3 = tf.constant(np.random.rand(2, 3, 1))
# encode_flag_and_match_box(boxes1, boxes2, boxes3)
