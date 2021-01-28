import tensorflow as tf
import numpy as np
from numpy.random import rand


def version():
    print(tf.__version__)
    print(tf.test.is_gpu_available())


def reshape():
    tensor1 = tf.constant(np.random.rand(2, 2, 3))
    tensor2 = tf.constant(np.random.rand(3, 3, 3))

    print(tensor1)
    print(tensor2)

    t1 = tf.reshape(tensor1, [-1, 3])
    t2 = tf.reshape(tensor2, [-1, 3])

    print(t1)
    print(t2)

    print(tf.concat([t1, t2], axis=0))
    tensor1 = tf.constant(np.arange(0, 96).reshape((2, 2, 2, 12)))
    print(tensor1)
    t2 = tf.reshape(tensor1, [2, 12, 4])
    print(t2)

    t3 = tf.reshape(t2, [12, 4])
    print(t3)


def loss():
    cls_gt = tf.constant(np.random.rand(3, 12, 1))
    anchor_flag = tf.constant(np.arange(0, 36).reshape(3, 12).astype(np.float64))
    cls_pred = tf.constant(np.random.rand(3, 12, 1))
    print(tf.keras.losses.binary_crossentropy(cls_gt, cls_pred) * anchor_flag)
    print(tf.keras.losses.binary_crossentropy(cls_gt, cls_pred))

    reg_gt = tf.constant(np.random.rand(2, 5, 4))
    anchor_flag = tf.constant(np.arange(0, 10).reshape(2, 5).astype(np.float64))
    reg_pred = tf.constant(np.random.rand(2, 5, 4))

    difference = reg_gt - reg_pred
    absolute_difference = tf.abs(difference)
    squared_difference = difference ** 2
    loss = tf.where(
        tf.less(absolute_difference, 0.5),
        0.5 * squared_difference,
        absolute_difference - 0.5,
    )
    print(loss)
    s = tf.reduce_sum(loss, axis=-1) * anchor_flag
    print(tf.reduce_sum(s))


def compare_and_gather():
    tensor1 = tf.constant(np.random.rand(4, 4))
    tensor2 = tf.constant(np.random.rand(4, 4))
    print(tensor1, tensor2)
    print(tensor1 > tensor2)
    x1, y1, x2, y2 = tf.split(tensor1, [1, 1, 1, 1], axis=1)
    print(x1, x2)

    not_tiny = tf.cast(x2 - x1 > 0.0002, tf.int32) * tf.cast(y2 - y1 > 0.0002, tf.int32)
    not_tiny = (x2 - x1 > 0.00002) & (y2 - y1 > 0.0002)
    print('&', not_tiny)
    idx = tf.reshape(not_tiny, [-1])
    print('idx', idx)
    print(tf.where(not_tiny), tf.where(idx))
    print(tf.gather(tensor1, tf.cast(tf.reshape(tf.where(idx), [-1]), tf.int32)))
    cast = tf.cast(x1 - x2 > 0.2, tf.int32)
    print(cast)
    print(cast == 1)
    print(tf.gather_nd(tensor1, tf.cast(x1 - x2 > 0.2, tf.int32) == 1))

    tensor_a = tf.Variable([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    tensor_b = tf.Variable([0, 1, 2], dtype=tf.int32)
    tensor_c = tf.Variable([[0], [0], [0]], dtype=tf.int32)
    print(tf.gather(tensor_a, tensor_b, batch_dims=-1))


def nms():
    boxes = tf.constant(np.random.rand(1, 10, 1, 4).astype(np.float32))
    scores = tf.constant(np.random.rand(1, 10, 1).astype(np.float32))
    box, scores, classes, detections = tf.image.combined_non_max_suppression(boxes, scores, 10, 3)
    print(boxes)
    print(box)


def roi_align_trick():
    roi_level = tf.constant(np.random.randint(0, 3, (2, 5)))
    print(roi_level)
    equal = tf.equal(roi_level, 2)
    where = tf.where(equal)
    print('where', where)
    box = tf.constant(np.random.rand(2, 5, 4))
    print(box)
    level_boxes = tf.gather_nd(box, where)
    # print(level_boxes)
    box_indices = tf.cast(where[:, 0], tf.int32)
    print(box_indices)


def box_area():
    b1 = tf.constant(rand(2, 5, 4))
    b2 = tf.constant(rand(2, 5, 4))

    print(b2[:, :, 2] - b1[:, :, 2])


def box_iou():
    boxes1 = tf.constant(rand(2, 5, 4))  # gt
    boxes2 = tf.constant(rand(2, 3, 4))
    # print(boxes1)
    b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 2),
                            [1, 1, 1, tf.shape(boxes2)[1]]),
                    [-1, tf.shape(boxes1)[1] * tf.shape(boxes2)[1], 4])
    b2 = tf.tile(boxes2, [1, tf.shape(boxes1)[1], 1])
    # print(b1)
    # print(boxes2)
    # print(b2)
    # 2. Compute intersections
    b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=2)
    b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=2)
    y1 = tf.maximum(b1_y1, b2_y1)
    x1 = tf.maximum(b1_x1, b2_x1)
    y2 = tf.minimum(b1_y2, b2_y2)
    x2 = tf.minimum(b1_x2, b2_x2)
    intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
    # 3. Compute unions
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area + b2_area - intersection
    # 4. Compute IoU and reshape to [boxes1, boxes2]
    iou = intersection / union
    overlaps = tf.reshape(iou, [-1, tf.shape(boxes1)[1], tf.shape(boxes2)[1]])
    return overlaps


def match_proposal(low_threshold=0.3, high_threshold=0.7):
    match_quality_matrix = box_iou()
    print(match_quality_matrix.shape)
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

    print(matches.shape)
    return matches


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
    for i in range(batch_size):
        inputs_slice = [x[i] for x in inputs]
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


if __name__ == '__main__':
    roi_level = tf.constant(np.random.randint(0, 100, (5,)))
    print(roi_level)
    equal = tf.equal(roi_level, 2)
    print(equal)
    where = tf.where(equal)
    print('where', where)
    print(tf.gather_nd(roi_level, where))
    roi = tf.random.shuffle(roi_level)
    print(roi)
    print(tf.slice(roi, [0], [2]))
