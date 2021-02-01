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
    """ encode grid anchors with 1,2,3...N,0,-1 and match gt box to each proposal
        gt_box shape:[batch_size, gt_num, 4]
        gt_label shape:[batch_size, gt_num, 1]
        proposal shape:[batch_size, proposal_num, 4]
        return
        clipped_matched_idxs shape:[batch_size, proposal_num]
        flags shape:[batch_size, proposal_num]
    """

    # if tf.size(gt_box) == 0:
    #     clipped_matched_idxs = tf.constant(np.zeros(tf.shape(proposal)))
    #     flags = tf.constant(np.zeros((tf.shape(proposal)[0],)))
    # else:
    match_quality_matrix = box_iou(gt_box, proposal)
    matched_idxs = match_proposal(match_quality_matrix)  # shape:[batch_size, proposal_num]
    # print(matched_idxs)
    # clipped_matched_idxs shape [batch_size, proposal_num]
    clipped_matched_idxs = tf.clip_by_value(matched_idxs, 0, 1000)
    flags = tf.gather(gt_label, clipped_matched_idxs, batch_dims=1)
    flags = tf.squeeze(flags, -1)
    flags = tf.cast(flags, tf.float64)

    bg_idx = tf.where(tf.equal(matched_idxs, -1))
    bg_update = tf.tile(tf.constant(np.array([0], dtype=np.float64)),
                        [tf.shape(bg_idx)[0]])
    flags = tf.tensor_scatter_nd_update(flags, bg_idx, bg_update)

    idx_to_discard = tf.where(tf.equal(matched_idxs, -2))
    discard_update = tf.tile(tf.constant(np.array([-1], dtype=np.float64)),
                             [tf.shape(idx_to_discard)[0]])
    flags = tf.tensor_scatter_nd_update(flags, idx_to_discard, discard_update)
    # print(clipped_matched_idxs.shape)
    # print(flags.shape)
    return clipped_matched_idxs, flags


def cal_reg_target(gt_box, proposal, weights=(1.0, 1.0, 1.0, 1.0)):
    """
    :param gt_box: shape [N, 4]
    :param proposal: shape [N, 4]
    :param weights:
    :return:
    """
    wx, wy, ww, wh = weights[0], weights[1], weights[2], weights[3]

    proposal_x1, proposal_y1, proposal_x2, proposal_y2 = tf.split(proposal, [1, 1, 1, 1], axis=1)

    gt_box_x1, gt_box_y1, gt_box_x2, gt_box_y2 = tf.split(gt_box, [1, 1, 1, 1], axis=1)

    ex_widths = proposal_x2 - proposal_x1
    ex_heights = proposal_y2 - proposal_y1
    ex_ctr_x = proposal_x1 + 0.5 * ex_widths
    ex_ctr_y = proposal_y1 + 0.5 * ex_heights

    gt_widths = gt_box_x2 - gt_box_x1
    gt_heights = gt_box_y2 - gt_box_y1
    gt_ctr_x = gt_box_x1 + 0.5 * gt_widths
    gt_ctr_y = gt_box_y1 + 0.5 * gt_heights

    targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = ww * tf.math.log(gt_widths / ex_widths)
    targets_dh = wh * tf.math.log(gt_heights / ex_heights)

    targets = tf.concat([targets_dx, targets_dy, targets_dw, targets_dh], axis=1)
    return targets


def balanced_sample(flags):
    """
    :param flags: shape [batch_size, N, 1]
    :param proposal: shape [batch_size, N, 4]
    :param gt_box: shape [batch_size, N, 4]
    :return:
    """
    print('flags shape' ,flags.shape)
    selected_sample = []
    batch_size, num_proposal = flags.get_shape().as_list()
    for i in range(batch_size):
        positive = tf.where(tf.greater_equal(flags[i], 0.1))
        negative = tf.where(tf.greater_equal(flags[i], 0))
        positive = tf.squeeze(positive, -1)
        negative = tf.squeeze(negative, -1)

        num_pos = 256
        # protect against not enough positive examples
        num_pos = tf.minimum(tf.size(positive), num_pos)
        num_neg = 512 - num_pos
        # protect against not enough negative examples
        num_neg = tf.minimum(tf.size(negative), num_neg)

        # randomly select positive and negative examples
        pos_idx_per_image = tf.slice(tf.random.shuffle(positive), [0], [num_pos])
        neg_idx_per_image = tf.slice(tf.random.shuffle(negative), [0], [num_neg])

        idx_per_image = tf.concat([pos_idx_per_image, neg_idx_per_image], axis=-1)
        idx_per_image_mask = tf.scatter_nd(tf.expand_dims(idx_per_image, -1),
                                           tf.ones([num_pos + num_neg]),
                                           tf.constant([num_proposal], dtype=tf.int64))
        selected_sample.append(idx_per_image_mask)
    select_sample_idx = tf.stack(selected_sample, axis=0)
    # _, sampled_indices = tf.nn.top_k(tf.cast(select_sample_idx, dtype=tf.int32), k=4, sorted=True)
    # sampled_indices_shape = tf.shape(sampled_indices)
    # batch_indices = (tf.expand_dims(tf.range(sampled_indices_shape[0]), axis=-1) *
    #                  tf.ones([1, sampled_indices_shape[-1]], dtype=tf.int32))
    # gather_nd_indices = tf.stack([batch_indices, sampled_indices], axis=-1)
    return select_sample_idx


def select_train_sample(gt_box, proposal, gt_label, batch_size=2):
    """
    :param gt_box: shape [batch_size, N, 4]
    :param proposal: shape [batch_size, M, 4]
    :param gt_label: shape [batch_size, N, 1]
    :return:
    """
    matched_idxs, flags = encode_flag_and_match_box(gt_box, proposal, gt_label)
    selected_sample_idxes = balanced_sample(flags)
    sampled_proposal, sampled_gt_box, sampled_gt_label, sampled_gt_idx = list(), list(), list(), list()

    for i in range(batch_size):
        selected_sample_idx = selected_sample_idxes[i]
        selected_sample_idx = tf.cast(selected_sample_idx, tf.int32)

        sampled_proposal.append(tf.gather(proposal[i], selected_sample_idx))
        sampled_gt_box.append(tf.gather(gt_box[i], selected_sample_idx))
        sampled_gt_label.append(tf.gather(gt_label[i], selected_sample_idx))
        sampled_gt_idx.append(tf.gather(matched_idxs[i], selected_sample_idx))

    sampled_proposal = tf.stack(sampled_proposal, axis=0).shape
    sampled_gt_box = tf.stack(sampled_gt_box, axis=0)
    sampled_gt_label = tf.stack(sampled_gt_label, axis=0)
    sampled_gt_idx = tf.stack(sampled_gt_idx, axis=0)
    return sampled_proposal, sampled_gt_box, sampled_gt_label, sampled_gt_idx


# encode_flag_and_match_box  test
# boxes1 = tf.constant(np.random.rand(2, 3, 4))
# boxes2 = tf.constant(np.random.rand(2, 5, 4))
# boxes3 = tf.constant(np.random.rand(2, 3, 1))
# encode_flag_and_match_box(boxes1, boxes2, boxes3)
# select_train_sample(boxes1, boxes2, boxes3)
# cal_reg_target  test
# boxes1 = np.random.rand(3, 4)
# boxes2 = np.random.rand(3, 4)
# cal_reg_target(boxes1, boxes2)

# balanced_sample test
# boxes1 = tf.constant(np.random.rand(2, 1000, 1))
# boxes2 = tf.constant(np.random.rand(2000, 4))
# boxes3 = tf.constant(np.random.rand(3, 4))
# out = balanced_sample(boxes1)
# _, sampled_indices = tf.nn.top_k(tf.cast(out, dtype=tf.int32),
#                                  k=4,
#                                  sorted=True)
# sampled_indices_shape = tf.shape(sampled_indices)
# batch_indices = (
#         tf.expand_dims(tf.range(sampled_indices_shape[0]), axis=-1) *
#         tf.ones([1, sampled_indices_shape[-1]], dtype=tf.int32))
# gather_nd_indices = tf.stack([batch_indices, sampled_indices], axis=-1)
#
# print(gather_nd_indices)

# boxes1 = tf.constant(np.random.rand(2, 10))
# boxes2 = tf.constant(np.array([[[0, 2],
#                                 [0, 2],
#                                 [0, 4],
#                                 [0, 4]],
#                                [[1, 6],
#                                 [1, 5],
#                                 [1, 8],
#                                 [1, 8]]]))
# out = tf.gather_nd(boxes1, boxes2)
# print(boxes1)
# print(out)
