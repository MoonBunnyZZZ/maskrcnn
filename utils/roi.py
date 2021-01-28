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
    print(clipped_matched_idxs.shape)
    print(flags.shape)
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


def balanced_sample(flags, proposal, gt_box):
    """

    :param flags:
    :param proposal:
    :param gt_box:
    :param ratio:
    :param num_anchors_sample:
    :return:
    """
    positive = tf.where(tf.greater_equal(flags, 0))
    negative = tf.where(tf.equal(flags, 0))
    print('tf.shape(positive)', tf.shape(positive))
    positive = tf.squeeze(positive, -1)
    negative = tf.squeeze(negative, -1)
    print('tf.shape(positive)', tf.shape(positive))

    num_pos = 256
    # protect against not enough positive examples
    num_pos = tf.minimum(tf.size(positive), num_pos)
    num_neg = 512 - num_pos
    # protect against not enough negative examples
    num_neg = tf.minimum(tf.size(negative), num_neg)

    # randomly select positive and negative examples
    print(num_pos)
    pos_idx_per_image = tf.slice(tf.random.shuffle(positive), [0], [num_pos])
    neg_idx_per_image = tf.slice(tf.random.shuffle(negative), [0], [num_neg])

    idx_per_image = tf.concat([pos_idx_per_image, neg_idx_per_image], axis=-1)
    labels_sampled = tf.gather(flags, idx_per_image)

    proposal_sampled = tf.gather(proposal, idx_per_image)
    gt_box_sampled = tf.gather(gt_box, idx_per_image)
    reg_target_sampled = cal_reg_target(gt_box_sampled, proposal_sampled)

    return labels_sampled, reg_target_sampled


def batch_slice(inputs, graph_fn, batch_size, names=None):
    """Splits inputs into slices and feeds each slice to a copy of the given
    computation graph and then combines the results. It allows you to run a
    graph on a batch of inputs even if the graph is written to support one
    instance only.

    inputs: list of tensors. All must have the same first dimension length
    graph_fn: A function that returns a TF tensor that's part of a graph.
    batch_size: number of slices to divide the data into.
    names: If provided, assigns names to the resulting tensors.
    """
    if not isinstance(inputs, list):
        inputs = [inputs]

    outputs = []
    for i in range(2):
        inputs_slice = [x[i] for x in inputs]
        print('slice', inputs_slice)
        output_slice = graph_fn(*inputs_slice)
        if not isinstance(output_slice, (tuple, list)):
            output_slice = [output_slice]
        outputs.append(output_slice)
    # Change outputs from a list of slices where each is
    # a list of outputs to a list of outputs and each has
    # a list of slices
    outputs = list(zip(*outputs))

    if names is None:
        names = [None] * len(outputs)

    result = [tf.stack(o, axis=0, name=n)
              for o, n in zip(outputs, names)]
    if len(result) == 1:
        result = result[0]

    return result


def select_train_sample(gt_box, proposal, gt_label):
    matched_idxs, flags = encode_flag_and_match_box(gt_box, proposal, gt_label)
    # out = Sample()([flags, proposal, gt_box])
    return matched_idxs, flags


class Sample(tf.keras.layers.Layer):
    def __init__(self):
        super(Sample, self).__init__()

    def call(self, inputs):
        flags, proposal, gt_box = inputs
        out = batch_slice([flags, proposal, gt_box],
                          lambda x, y, z: balanced_sample(x, y, z),
                          2)
        return out


# encode_flag_and_match_box  test
# boxes1 = tf.constant(np.random.rand(2, 3, 4))
# boxes2 = tf.constant(np.random.rand(2, 5, 4))
# boxes3 = tf.constant(np.random.rand(2, 3, 1))
# encode_flag_and_match_box(boxes1, boxes2, boxes3)

# cal_reg_target  test
# boxes1 = np.random.rand(3, 4)
# boxes2 = np.random.rand(3, 4)
# cal_reg_target(boxes1, boxes2)

# balanced_sample test
# boxes1 = tf.constant(np.random.rand(2000, ))
# boxes2 = tf.constant(np.random.rand(2000, 4))
# boxes3 = tf.constant(np.random.rand(3, 4))
# balanced_sample(boxes1, boxes2, boxes3)

inputs1 = tf.keras.layers.Input(shape=(None, None), name='image')
inputs2 = tf.keras.layers.Input(shape=(None, None), name='image')
inputs3 = tf.keras.layers.Input(shape=(None, 4), name='image')

s = Sample()
res = s([inputs1, inputs2, inputs3])
m = tf.keras.Model(inputs=[inputs1, inputs2, inputs3], outputs=res)
m.summary()
