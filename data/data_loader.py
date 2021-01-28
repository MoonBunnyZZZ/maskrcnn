import cv2
import numpy as np

from utils import box_ops
from utils.anchor import set_cell_anchors, set_grid_anchors
from data.dataset import Dataset


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
    num_pos = min(positive.size(), num_pos)
    num_neg = num_anchors_sample - num_pos
    # protect against not enough negative examples
    num_neg = min(negative.size(), num_neg)

    # randomly select positive and negative examples
    perm1 = np.random.permutation(positive.size())[:num_pos]
    perm2 = np.random.permutation(negative.size())[:num_neg]

    pos_idx_per_image = positive[perm1]
    neg_idx_per_image = negative[perm2]

    # create binary mask from indices
    pos_idx_per_image_mask = np.zeros_like(flags)
    neg_idx_per_image_mask = np.zeros_like(flags)

    pos_idx_per_image_mask[pos_idx_per_image] = 1
    neg_idx_per_image_mask[neg_idx_per_image] = 1

    return pos_idx_per_image_mask, neg_idx_per_image_mask


def rpn_target(anchor, gt_box):
    """
    :param anchor: np array, shape (N, 4)
    :param gt_box: np array, shape (M, 4)
    :return: (N, ), (N, 4)
    """
    anchor = np.concatenate(anchor, axis=0)
    flags, matched_gt_boxes = encode_flag_and_match_box(gt_box, anchor)
    regression_targets = cal_reg_target(matched_gt_boxes, anchor)

    return flags, regression_targets


def transform_matrix(w, h,
                     zoom_ratio=(0.8, 1.2),
                     shift_ratio=(-0.2, 0.2),
                     rotation_degree=(-20, 20)):
    t1 = np.array([[1, 0, -w // 2], [0, 1, -h // 2], [0, 0, 1]])
    t2 = np.array([[1, 0, w // 2], [0, 1, h // 2], [0, 0, 1]])

    zx, zy = np.random.uniform(zoom_ratio[0], zoom_ratio[1], 2)
    zoom_matrix = np.array([[zx, 0, 0], [0, zy, 0], [0, 0, 1]])

    # shear = np.random.uniform(-0.2, 0.2)
    # shear_martrix = np.array([[1, -np.sin(shear), 0], [0, np.cos(shear), 0], [0, 0, 1]])

    tx, ty = np.random.uniform(-shift_ratio[0], shift_ratio[1], 2)
    shift_matrix = np.array([[1, 0, tx * w], [0, 1, ty * h], [0, 0, 1]])

    theta = np.pi / 180 * np.random.uniform(rotation_degree[0], rotation_degree[1])
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

    m = np.matmul(zoom_matrix, shift_matrix)
    m = np.matmul(rotation_matrix, m)

    m = (np.matmul(t2, np.matmul(m, t1)))
    return m[0:2, 0:3]


def augment_data(image, mask):
    h, w = image.shape
    m = transform_matrix(w, h)
    image = cv2.warpAffine(image, m, (w, h))
    mask = cv2.warpAffine(mask, m, (w, h))
    if np.random.rand(1)[0]:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)
    return image, mask


def cal_bbox_from_mask(mask):
    """calculate lt rb"""
    matrix = mask > 0
    y, x = np.nonzero(matrix)
    y1 = np.min(y)
    x1 = np.min(x)
    y2 = np.max(y)
    x2 = np.max(x)
    return np.array([[x1, y1, x2, y2]])


def data_generator(h5_path,
                   sizes, aspect_ratios,
                   grid_sizes, strides, batch_size,
                   positive_sample_ratio, num_anchors_sample):
    brats = Dataset(h5_path)
    cell_anchors = set_cell_anchors(sizes, aspect_ratios)
    anchors = set_grid_anchors(grid_sizes, strides, cell_anchors)

    item_idx = 0
    batch_images, batch_flags, batch_regs = list(), list(), list()
    batch_pos_sample_idx, batch_neg_sample_idx = list(), list()
    while True:
        image, gt_mask, gt_cls = brats.load(item_idx)
        image, gt_mask = augment_data(image, gt_mask)
        gt_box = cal_bbox_from_mask(gt_mask)

        flag, rpn_reg = rpn_target(anchors, gt_box)
        pos_idx, neg_idx = balanced_sample(flag, positive_sample_ratio, num_anchors_sample)

        batch_images.append(image)
        batch_flags.append(flag)
        batch_regs.append(rpn_reg)
        batch_pos_sample_idx.append(pos_idx)
        batch_neg_sample_idx.append(neg_idx)

        if (item_idx + 1) % batch_size == 0:
            # yield np.array(batch_images), np.array(batch_flags), np.array(batch_regs)
            batch_images.clear(), batch_flags.clear(), batch_regs.clear()
        if item_idx > len(brats):
            item_idx = 0


def dummy_generator():
    item_idx = 0
    batch_images, batch_flags, batch_regs = list(), list(), list()
    batch_pos_sample_idx, batch_neg_sample_idx = list(), list()
    while True:
        batch_images.append(np.random.rand(448, 448, 3))
        batch_flags.append(np.random.randint(0, 2, (12495, 1)).astype(np.float64))
        batch_regs.append(np.random.rand(12495, 4))
        batch_pos_sample_idx.append(np.random.randint(0, 2, (12495, 1)).astype(np.float64))
        batch_neg_sample_idx.append(np.random.randint(0, 2, (12495, 1)).astype(np.float64))

        if (item_idx + 1) % 1 == 0:
            inputs = {"image": np.array(batch_images),
                      "rpn_cls_gt": np.array(batch_flags),
                      "rpn_reg_gt": np.array(batch_regs),
                      "rpn_pos_sample_idx": np.array(batch_pos_sample_idx),
                      "rpn_sample_idx": np.array(batch_neg_sample_idx)}
            targets = {}
            yield inputs, targets
            batch_images.clear(), batch_flags.clear(), batch_regs.clear()
            batch_pos_sample_idx.clear(), batch_neg_sample_idx.clear()
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
    pass
