import numpy as np


def generate_anchors(scales=(32,), aspect_ratios=(0.5, 1, 2), dtype=np.float32):
    """generate anchors of one scale at all aspect_ratios"""
    scales = np.array(scales)
    aspect_ratios = np.array(aspect_ratios, dtype=dtype)
    h_ratios = np.sqrt(aspect_ratios)
    w_ratios = 1 / h_ratios

    ws = (w_ratios[:, None] * scales[None, :]).reshape(-1)
    hs = (h_ratios[:, None] * scales[None, :]).reshape(-1)

    base_anchors = np.stack([-ws, -hs, ws, hs], axis=1) / 2
    return base_anchors


def set_cell_anchors(sizes, aspect_ratios):
    cell_anchors = [generate_anchors(sizes, aspect_ratios) for sizes, aspect_ratios in zip(sizes, aspect_ratios)]
    return cell_anchors


def set_grid_anchors(grid_sizes, strides, cell_anchors):
    """put cell anchor to fpn feature map correspondingly"""
    anchors = []
    assert cell_anchors is not None

    for size, stride, base_anchors in zip(grid_sizes, strides, cell_anchors):
        grid_height, grid_width = size
        stride_height, stride_width = stride

        # For output anchor, compute [x_center, y_center, x_center, y_center]
        shifts_x = np.arange(0, grid_width) * stride_width
        shifts_y = np.arange(0, grid_height) * stride_height
        shift_x, shift_y = np.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        shifts = np.stack((shift_x, shift_y, shift_x, shift_y), axis=1)

        # For every (base anchor, output anchor) pair,
        # offset each zero-centered base anchor by the center of the output anchor.
        anchors.append(
            (shifts.reshape((-1, 1, 4)) + base_anchors.reshape((1, -1, 4))).reshape(-1, 4)
        )

    return anchors


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


def match_proposal(match_quality_matrix, low_threshold=0.3, high_threshold=0.7, allow_low_quality_matches=False):
    matched_vals = np.max(match_quality_matrix, axis=0)
    matches = np.argmax(match_quality_matrix, axis=0)
    all_matches = matches.copy() if allow_low_quality_matches else None

    below_low_threshold = matched_vals < low_threshold
    between_thresholds = (matched_vals >= low_threshold) & (matched_vals < high_threshold)
    matches[below_low_threshold] = -1
    matches[between_thresholds] = -2

    if allow_low_quality_matches:
        matches = set_low_quality_matches(matches, all_matches, match_quality_matrix)

    return matches


def set_low_quality_matches(matches, all_matches, match_quality_matrix):
    highest_quality_foreach_gt = match_quality_matrix.max(axis=1)

    x, y = np.nonzero(match_quality_matrix == highest_quality_foreach_gt[:, None])
    gt_pred_pairs_of_highest_quality = np.stack((x, y), axis=1)

    pred_inds_to_update = gt_pred_pairs_of_highest_quality[:, 1]
    matches[pred_inds_to_update] = all_matches[pred_inds_to_update]
    return matches


def cal_reg_target(gt_boxes, grid_anchors, weights=(1.0, 1.0, 1.0, 1.0)):
    wx, wy, ww, wh = weights[0], weights[1], weights[2], weights[3]
    anchors_x1 = np.expand_dims(grid_anchors[:, 0], axis=1)
    anchors_y1 = np.expand_dims(grid_anchors[:, 1], axis=1)
    anchors_x2 = np.expand_dims(grid_anchors[:, 2], axis=1)
    anchors_y2 = np.expand_dims(grid_anchors[:, 3], axis=1)

    gt_boxes_x1 = np.expand_dims(gt_boxes[:, 0], axis=1)
    gt_boxes_y1 = np.expand_dims(gt_boxes[:, 1], axis=1)
    gt_boxes_x2 = np.expand_dims(gt_boxes[:, 2], axis=1)
    gt_boxes_y2 = np.expand_dims(gt_boxes[:, 3], axis=1)

    ex_widths = anchors_x2 - anchors_x1
    ex_heights = anchors_y2 - anchors_y1
    ex_ctr_x = anchors_x1 + 0.5 * ex_widths
    ex_ctr_y = anchors_y1 + 0.5 * ex_heights

    gt_widths = gt_boxes_x2 - gt_boxes_x1
    gt_heights = gt_boxes_y2 - gt_boxes_y1
    gt_ctr_x = gt_boxes_x1 + 0.5 * gt_widths
    gt_ctr_y = gt_boxes_y1 + 0.5 * gt_heights

    targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths + 1e-8
    targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights + 1e-8
    targets_dw = ww * np.log(gt_widths / ex_widths + 1e-8)
    targets_dh = wh * np.log(gt_heights / ex_heights + 1e-8)

    targets = np.concatenate((targets_dx, targets_dy, targets_dw, targets_dh), axis=1)
    return targets


def encode_flag_and_match_box(gt_box, anchor):
    """encode grid anchors with 1,0,-1 and match gt box to each anchor"""
    if gt_box.size == 0:
        matched_gt_boxes = np.zeros(anchor.shape)
        flags = np.zeros((anchor.shape[0],))
    else:
        match_quality_matrix = box_iou(gt_box, anchor)
        matched_idxs = match_proposal(match_quality_matrix)

        matched_gt_boxes = gt_box[matched_idxs.clip(min=0)]

        flags = matched_idxs >= 0
        flags = flags.astype(np.float32)

        bg_indices = matched_idxs == -1
        flags[bg_indices] = 0.0

        inds_to_discard = matched_idxs == -2
        flags[inds_to_discard] = -1.0
    return flags, matched_gt_boxes


def balanced_sample(flags, ratio, num_anchors_sample):
    positive = np.nonzero(flags >= 1)[0]
    negative = np.nonzero(flags == 0)[0]

    num_pos = int(num_anchors_sample * ratio)
    # protect against not enough positive examples
    num_pos = min(positive.size, num_pos)
    num_neg = num_anchors_sample - num_pos
    # protect against not enough negative examples
    num_neg = min(negative.size, num_neg)

    # randomly select positive and negative examples
    perm1 = np.random.permutation(positive.size)[:num_pos]
    perm2 = np.random.permutation(negative.size)[:num_neg]

    pos_idx = positive[perm1]
    neg_idx = negative[perm2]
    return pos_idx, neg_idx


def rpn_target(anchor, gt_box, ratio, sample_num):
    """
    anchor: np array, shape (N, 4)
    gt_box: np array, shape (M, 4)
    flags, regression_targets (N, ), (N, 4)
    """
    anchor = np.concatenate(anchor, axis=0)
    flags, matched_gt_boxes = encode_flag_and_match_box(gt_box, anchor)
    reg_targets = cal_reg_target(matched_gt_boxes, anchor)
    pos_idx, neg_idx = balanced_sample(flags, ratio, sample_num)

    new_flags = np.zeros_like(flags) - 1.0
    new_flags[pos_idx] = 1
    new_flags[neg_idx] = 0

    new_targets = np.zeros_like(reg_targets)
    new_targets[pos_idx] = reg_targets[pos_idx]
    return flags, reg_targets


def cal_bbox_from_mask(mask):
    """calculate lt rb"""
    matrix = mask > 0
    y, x = np.nonzero(matrix)
    y1 = np.min(y)
    x1 = np.min(x)
    y2 = np.max(y)
    x2 = np.max(x)
    return np.array([[x1, y1, x2, y2]])


# def data_generator(h5_path,
#                    sizes, aspect_ratios,
#                    grid_sizes, strides, batch_size,
#                    positive_sample_ratio, num_anchors_sample):
#     brats = Dataset(h5_path)
#     cell_anchors = set_cell_anchors(sizes, aspect_ratios)
#     anchors = set_grid_anchors(grid_sizes, strides, cell_anchors)
#
#     item_idx = 0
#     batch_images, batch_flags, batch_regs = list(), list(), list()
#     batch_pos_sample_idx, batch_neg_sample_idx = list(), list()
#     while True:
#         image, gt_mask, gt_cls = brats.load(item_idx)
#         image, gt_mask = augment_data(image, gt_mask)
#         gt_box = cal_bbox_from_mask(gt_mask)
#
#         flag, rpn_reg = rpn_target(anchors, gt_box)
#         pos_idx, neg_idx = balanced_sample(flag, positive_sample_ratio, num_anchors_sample)
#
#         batch_images.append(image)
#         batch_flags.append(flag)
#         batch_regs.append(rpn_reg)
#         batch_pos_sample_idx.append(pos_idx)
#         batch_neg_sample_idx.append(neg_idx)
#
#         if (item_idx + 1) % batch_size == 0:
#             # yield np.array(batch_images), np.array(batch_flags), np.array(batch_regs)
#             batch_images.clear(), batch_flags.clear(), batch_regs.clear()
#         if item_idx > len(brats):
#             item_idx = 0


def dummy_generator():
    item_idx = 0
    cell_anchors = set_cell_anchors(((32,), (64,), (128,)), ((0.5, 1, 2),) * 3)
    anchors = set_grid_anchors([[28, 28], [14, 14], [7, 7]],
                               [[8, 8], [16, 16], [32, 32]], cell_anchors)
    batch_images, batch_gt_box, batch_gt_cls, batch_rpn_reg, batch_rpn_cls = list(), list(), list(), list(), list()
    while True:
        image = np.random.rand(224, 224, 3)
        batch_images.append(image)

        gt_box = np.random.rand(3, 4)
        batch_gt_box.append(gt_box)
        gt_cls = np.random.rand(3, 1)
        batch_gt_cls.append(gt_cls)

        rpn_cls, rpn_reg = rpn_target(anchors, gt_box, 0.5, 256)
        batch_rpn_reg.append(rpn_reg)
        batch_rpn_cls.append(rpn_cls)
        if (item_idx + 1) % 1 == 0:
            batch_data = {'image': np.array(batch_images),
                          'gt_box': np.array(batch_gt_box),
                          'gt_cls': np.array(batch_gt_cls),
                          'rpn_reg': np.array(batch_rpn_reg),
                          'rpn_cls': np.array(batch_rpn_cls),
                          'anchor': np.concatenate(anchors, axis=0)}
            yield batch_data
            # yield np.array(batch_images), np.array(batch_gt_box), np.array(batch_gt_cls), \
            #       np.array(batch_rpn_reg), np.array(batch_rpn_cls), np.concatenate(anchors, axis=0)

            batch_images.clear()
            batch_gt_box.clear()
            batch_gt_cls.clear()
            batch_rpn_reg.clear()
            batch_rpn_cls.clear()
            item_idx = 0


if __name__ == '__main__':
    # test match_proposal
    # matrix = np.random.rand(2, 5)
    # print(matrix)
    # match = match_proposal(matrix, allow_low_quality_matches=False)
    # print(match)
    # match = match_proposal(matrix, allow_low_quality_matches=True)
    # print(match)

    # test encode_flag_and_match_box
    # anchor = np.array([[1, 1, 3, 3],
    #                    [2, 2, 5, 5]])
    # r_box = np.array([[1.5, 0.5, 4.5, 2.5],
    #                   [3, 3, 5.5, 6]])
    # print(encode_boxes(r_box, anchor))
    # dummy = dummy_generator()
    # for i, (images, gt_box, gt_cls, rpn_reg, rpn_cls, anchor) in enumerate(dummy):
    #     print(i, images.shape, gt_box.shape, gt_cls.shape, rpn_reg.shape, rpn_cls.shape, anchor.shape)

    cell_anchors = set_cell_anchors(((32,), (64,), (128,)), ((0.5, 1, 2),) * 3)
    anchors = set_grid_anchors([[28, 28], [14, 14], [7, 7]], [[8, 8], [16, 16], [32, 32]], cell_anchors)
    for i in anchors:
        print(i.shape)
    pass
