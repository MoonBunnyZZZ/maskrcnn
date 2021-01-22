import numpy as np
from utils import box_ops


def generate_anchors(scales=(32,), aspect_ratios=(0.5, 1, 2), dtype=np.float32):
    """generate anchors of one scale at all aspect_ratios"""
    scales = np.array(scales)
    aspect_ratios = np.array(aspect_ratios, dtype=dtype)
    h_ratios = np.sqrt(aspect_ratios)
    w_ratios = 1 / h_ratios

    ws = (w_ratios[:, None] * scales[None, :]).reshape(-1)
    hs = (h_ratios[:, None] * scales[None, :]).reshape(-1)

    base_anchors = np.stack([-ws, -hs, ws, hs], axis=1) / 2
    return np.round(base_anchors)


def set_cell_anchors(sizes, aspect_ratios):
    cell_anchors = [generate_anchors(sizes, aspect_ratios) for sizes, aspect_ratios in zip(sizes, aspect_ratios)]
    return cell_anchors


def grid_anchors(grid_sizes, strides, cell_anchors):
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


def assign_targets_to_anchors(self, anchors, targets):
    labels = []
    matched_gt_boxes = []
    for anchors_per_image, targets_per_image in zip(anchors, targets):
        gt_boxes = targets_per_image["boxes"]

        if gt_boxes.numel() == 0:
            # Background image (no gt box, negative example)
            device = anchors_per_image.device
            matched_gt_boxes_per_image = np.zeros(anchors_per_image.shape)
            labels_per_image = np.zeros((anchors_per_image.shape[0],))
        else:
            match_quality_matrix = box_ops.box_iou(gt_boxes, anchors_per_image)
            matched_idxs = self.proposal_matcher(match_quality_matrix)
            # get the targets corresponding GT for each proposal
            # NB: need to clamp the indices because we can have a single
            # GT in the image, and matched_idxs can be -2, which goes
            # out of bounds
            matched_gt_boxes_per_image = gt_boxes[matched_idxs.clamp(min=0)]

            labels_per_image = matched_idxs >= 0
            labels_per_image = labels_per_image.to(dtype=np.float32)

            # Background (negative examples)
            bg_indices = matched_idxs == self.proposal_matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_indices] = 0.0

            # discard indices that are between thresholds
            inds_to_discard = matched_idxs == self.proposal_matcher.BETWEEN_THRESHOLDS
            labels_per_image[inds_to_discard] = -1.0

        labels.append(labels_per_image)
        matched_gt_boxes.append(matched_gt_boxes_per_image)
    return labels, matched_gt_boxes


if __name__ == '__main__':
    cell = set_cell_anchors(((32,), (64,)), ((0.5, 1, 2),) * 2)
    grid = grid_anchors([[4, 4], [2, 2]], [[2, 2], [4, 4]], cell)
    print(grid)
