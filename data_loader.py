import numpy as np
from utils import box_ops


class Brats:
    def __init__(self, file_dir):
        self.file_dir = file_dir


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

    targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = ww * np.log(gt_widths / ex_widths)
    targets_dh = wh * np.log(gt_heights / ex_heights)

    targets = np.concatenate((targets_dx, targets_dy, targets_dw, targets_dh), axis=1)
    return targets


def encode_flag_and_match_box(gt_box, anchor):
    """encode grid anchors with 1,0,-1 and match gt box to each anchor"""
    if gt_box.size == 0:
        matched_gt_boxes = np.zeros(anchor.shape)
        flags = np.zeros((anchor.shape[0],))
    else:
        match_quality_matrix = box_ops.box_iou(gt_box, anchor)
        matched_idxs = match_proposal(match_quality_matrix)

        matched_gt_boxes = gt_box[matched_idxs.clamp(min=0)]

        flags = matched_idxs >= 0
        flags = flags.to(dtype=np.float32)

        bg_indices = matched_idxs == -1
        flags[bg_indices] = 0.0

        inds_to_discard = matched_idxs == -2
        flags[inds_to_discard] = -1.0
    return flags, matched_gt_boxes


def rpn_target(anchor, gt_box):
    """
    :param anchor: np array, shape (N, 4)
    :param gt_box: np array, shape (M, 4)
    :return: (N, ), (N, 4)
    """
    flags, matched_gt_boxes = encode_flag_and_match_box(gt_box, anchor)
    regression_targets = cal_reg_target(matched_gt_boxes, anchor)

    return flags, regression_targets


def data_generator():
    pass


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
    pass
