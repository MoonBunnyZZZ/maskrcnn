import numpy as np
import tensorflow as tf


def box_area(boxes):
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.
    Arguments:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format
    Returns:
        area (Tensor[N]): area for each box
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1, boxes2):
    """
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clip(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou


def clip_box(box, h, w, name):
    x1, y1, x2, y2 = tf.split(box, [1, 1, 1, 1], axis=2)
    x1 = tf.clip_by_value(x1, 0, w, name='{}_clip_x1'.format(name))
    x2 = tf.clip_by_value(x2, 0, w, name='{}_clip_y1'.format(name))
    y1 = tf.clip_by_value(y1, 0, h, name='{}_clip_x2'.format(name))
    y2 = tf.clip_by_value(y2, 0, h, name='{}_clip_y2'.format(name))
    box_clipped = tf.concat([x1, y1, x2, y2], axis=2, name='{}_clip_box'.format(name))

    return box_clipped


def remove_tiny(box, cls, length_thresh):
    x1, y1, x2, y2 = tf.split(box, [1, 1, 1, 1], axis=2)
    not_tiny = (x2 - x1 > length_thresh) & (y2 - y1 > length_thresh)
    print(not_tiny)
    print(tf.where(not_tiny))
    out=tf.gather_nd(box, tf.where(not_tiny)[:,:-1])
    print(out.shape)
    idx = tf.reshape(not_tiny, [-1])
    idx = tf.cast(tf.reshape(tf.where(idx), [-1]), tf.int32)
    # print(box)
    print(idx)
    box = tf.gather(box, idx)
    cls = tf.gather(cls, idx)
    return box, cls


def nms(box, cls, max_out_size, iou_threshold):
    box = tf.expand_dims(box, 2)
    cls = tf.expand_dims(cls, -1)
    boxes, scores, classes, detections = tf.image.combined_non_max_suppression(box, cls,
                                                                               max_output_size_per_class=max_out_size,
                                                                               max_total_size=max_out_size,
                                                                               iou_threshold=iou_threshold)
    return boxes


def reg_to_box(reg, anchor, batch_size=1):
    boxes = list()
    for i in range(batch_size):
        xa1, ya1, xa2, ya2 = tf.split(anchor, [1, 1, 1, 1], axis=-1)
        tx, ty, tw, th = tf.split(reg[i], [1, 1, 1, 1], axis=-1)
        # tx = tf.clip_by_value()  whether or not need clip

        wa = xa2 - xa1
        ha = ya2 - ya1
        xa = xa1 + wa * tf.convert_to_tensor(0.5)
        ya = ya1 + ha * tf.convert_to_tensor(0.5)

        x1 = xa - wa * tx
        x2 = xa + wa * tx
        y1 = ya - ha * ty
        y2 = ya + ha * ty

        boxes.append(tf.concat([x1, y1, x2, y2], axis=-1))
    return tf.stack(boxes, axis=0)


def top_k_boxes(boxes, scores, k):
    """Sort and select top k boxes according to the scores.

    Args:
      boxes: a tensor of shape [batch_size, N, 4] representing the coordiante of
        the boxes. N is the number of boxes per image.
      scores: a tensor of shsape [batch_size, N] representing the socre of the
        boxes.
      k: an integer or a tensor indicating the top k number.

    Returns:
      selected_boxes: a tensor of shape [batch_size, k, 4] representing the
        selected top k box coordinates.
      selected_scores: a tensor of shape [batch_size, k] representing the selected
        top k box scores.
    """
    with tf.name_scope('top_k_boxes'):
        selected_scores, top_k_indices = tf.nn.top_k(scores, k=k, sorted=True)

        batch_size, _ = scores.get_shape().as_list()
        if batch_size == 1:
            selected_boxes = tf.squeeze(
                tf.gather(boxes, top_k_indices, axis=1), axis=1)
        else:
            top_k_indices_shape = tf.shape(top_k_indices)
            batch_indices = (
                    tf.expand_dims(tf.range(top_k_indices_shape[0]), axis=-1) *
                    tf.ones([1, top_k_indices_shape[-1]], dtype=tf.int32))
            gather_nd_indices = tf.stack([batch_indices, top_k_indices], axis=-1)
            selected_boxes = tf.gather_nd(boxes, gather_nd_indices)

        return selected_boxes, selected_scores
