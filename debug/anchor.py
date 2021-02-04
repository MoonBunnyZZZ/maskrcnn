import collections
import numpy as np
import tensorflow as tf


class Anchor:
    """Anchor class for anchor-based object detectors."""

    def __init__(self, min_level, max_level, num_scales, aspect_ratios,
                 anchor_size, image_size):
        """Constructs multiscale anchors.

        Args:
          min_level: integer number of minimum level of the output feature pyramid.
          max_level: integer number of maximum level of the output feature pyramid.
          num_scales: integer number representing intermediate scales added on each
            level. For instances, num_scales=2 adds one additional intermediate
            anchor scales [2^0, 2^0.5] on each level.
          aspect_ratios: list of float numbers representing the aspect ratio anchors
            added on each level. The number indicates the ratio of width to height.
            For instances, aspect_ratios=[1.0, 2.0, 0.5] adds three anchors on each
            scale level.
          anchor_size: float number representing the scale of size of the base
            anchor to the feature stride 2^level.
          image_size: a list of integer numbers or Tensors representing [height,
            width] of the input image size.The image_size should be divisible by the
            largest feature stride 2^max_level.
        """
        self.min_level = min_level
        self.max_level = max_level
        self.num_scales = num_scales
        self.aspect_ratios = aspect_ratios
        self.anchor_size = anchor_size
        self.image_size = image_size
        self.boxes = self._generate_boxes()

    def _generate_boxes(self):
        """Generates multiscale anchor boxes.

        Returns:
          a Tensor of shape [N, 4], represneting anchor boxes of all levels
          concatenated together.
        """
        boxes_all = []
        for level in range(self.min_level, self.max_level + 1):
            boxes_l = []
            for scale in range(self.num_scales):
                for aspect_ratio in self.aspect_ratios:
                    stride = 2 ** level
                    intermediate_scale = 2 ** (scale / float(self.num_scales))
                    base_anchor_size = self.anchor_size * stride * intermediate_scale
                    aspect_x = aspect_ratio ** 0.5
                    aspect_y = aspect_ratio ** -0.5
                    half_anchor_size_x = base_anchor_size * aspect_x / 2.0
                    half_anchor_size_y = base_anchor_size * aspect_y / 2.0
                    x = tf.range(stride / 2, self.image_size[1], stride)
                    y = tf.range(stride / 2, self.image_size[0], stride)
                    xv, yv = tf.meshgrid(x, y)
                    xv = tf.cast(tf.reshape(xv, [-1]), dtype=tf.float32)
                    yv = tf.cast(tf.reshape(yv, [-1]), dtype=tf.float32)
                    # Tensor shape Nx4.
                    boxes = tf.stack([
                        yv - half_anchor_size_y, xv - half_anchor_size_x,
                        yv + half_anchor_size_y, xv + half_anchor_size_x
                    ],
                        axis=1)
                    boxes_l.append(boxes)
            # Concat anchors on the same level to tensor shape NxAx4.
            boxes_l = tf.stack(boxes_l, axis=1)
            boxes_l = tf.reshape(boxes_l, [-1, 4])
            boxes_all.append(boxes_l)
        return tf.concat(boxes_all, axis=0)

    def unpack_labels(self, labels):
        """Unpacks an array of labels into multiscales labels."""
        # unpacked_labels = collections.OrderedDict()
        unpacked_labels = list()
        count = 0
        for level in range(self.min_level, self.max_level + 1):
            feat_size_y = tf.cast(self.image_size[0] / 2 ** level, tf.int32)
            feat_size_x = tf.cast(self.image_size[1] / 2 ** level, tf.int32)
            steps = feat_size_y * feat_size_x * self.anchors_per_location
            unpacked_labels.append(tf.reshape(labels[count:count + steps],
                                              [feat_size_y*feat_size_x*len(self.aspect_ratios), -1]))
            count += steps
        return unpacked_labels

    @property
    def anchors_per_location(self):
        return self.num_scales * len(self.aspect_ratios)

    @property
    def multilevel_boxes(self):
        return self.unpack_labels(self.boxes)


class AnchorLabeler:
    def __init__(self, anchor, sample_num=512, pos_sample_ratio=0.5, low_threshold=0.3, high_threshold=0.7):
        self.anchor = anchor
        self.sample_num = sample_num
        self.pos_sample_ratio = pos_sample_ratio
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def encode_flag_and_match_box(self, gt_box):
        """ encode grid anchors with 1,2,3...N,0,-1 and match gt box to each proposal
            gt_box shape:[batch_size, gt_num, 4]
            anchor shape:[batch_size, anchor_num, 4]
            return
            matched_idxs shape:[batch_size, anchor_num]
            labels shape:[batch_size, anchor_num]
        """

        match_quality_matrix = self.box_iou(gt_box, self.anchor)
        matched_idxs = self.match_proposal(match_quality_matrix)  # shape:[batch_size, proposal_num]

        # matched_idxs shape [batch_size, proposal_num]
        matched_idxs = tf.clip_by_value(matched_idxs, 0, 1000)

        labels = tf.where(tf.greater_equal(matched_idxs, 0),
                          tf.ones_like(matched_idxs),
                          tf.zeros_like(matched_idxs))
        labels = tf.where(tf.equal(matched_idxs, -1),
                          tf.zeros_like(labels),
                          labels)
        labels = tf.where(tf.greater_equal(matched_idxs, -2),
                          -tf.ones_like(labels),
                          labels)
        return matched_idxs, labels

    @staticmethod
    def box_iou(boxes1, boxes2):
        b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, -1),
                                [1, 1, tf.shape(boxes2)[0]]),
                        [tf.shape(boxes1)[0] * tf.shape(boxes2)[0], 4])
        b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])

        b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=-1)
        b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=-1)
        y1 = tf.maximum(b1_y1, b2_y1)
        x1 = tf.maximum(b1_x1, b2_x1)
        y2 = tf.minimum(b1_y2, b2_y2)
        x2 = tf.minimum(b1_x2, b2_x2)
        intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)

        b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
        b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
        union = b1_area + b2_area - intersection

        iou = intersection / union
        overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])
        return overlaps

    def match_proposal(self, match_quality_matrix):
        """
            match_quality_matrix shape:[batch_size, gt_num, anchor_num]
            return shape:[batch_size, anchor_num]
        """
        matched_vals = tf.reduce_max(match_quality_matrix, axis=0)
        matches = tf.argmax(match_quality_matrix, axis=0)

        below_low_threshold = matched_vals < self.low_threshold
        between_thresholds = (matched_vals >= self.low_threshold) & (matched_vals < self.high_threshold)

        below_idx = tf.where(below_low_threshold)
        below_update = tf.tile(tf.constant(np.array([-1], dtype=np.int64)),
                               [tf.shape(below_idx)[0]])
        matches = tf.tensor_scatter_nd_update(matches, below_idx, below_update)

        between_idx = tf.where(between_thresholds)
        between_update = tf.tile(tf.constant(np.array([-2], dtype=np.int64)),
                                 [tf.shape(between_idx)[0]])
        matches = tf.tensor_scatter_nd_update(matches, between_idx, between_update)

        return matches

    def balanced_sample(self, flags):
        """
        :param flags: shape [batch_size, N, 1]
        :return:
        """
        selected_sample = []
        num_proposal = flags.get_shape().as_list()[0]
        # for i in range(batch_size):
        positive = tf.where(tf.greater_equal(flags, 1))
        negative = tf.where(tf.equal(flags, 0))
        positive = tf.squeeze(positive, -1)
        negative = tf.squeeze(negative, -1)

        num_pos = int(self.sample_num * self.pos_sample_ratio)
        # protect against not enough positive examples
        num_pos = tf.minimum(tf.size(positive), num_pos)
        num_neg = self.sample_num - num_pos
        # protect against not enough negative examples
        num_neg = tf.minimum(tf.size(negative), num_neg)

        # randomly select positive and negative examples
        pos_idx_per_image = tf.slice(tf.random.shuffle(positive), [0], [num_pos])
        neg_idx_per_image = tf.slice(tf.random.shuffle(negative), [0], [num_neg])

        idx_per_image = tf.concat([pos_idx_per_image, neg_idx_per_image], axis=-1)
        idx_per_image_mask = tf.scatter_nd(tf.expand_dims(idx_per_image, -1),
                                           tf.ones([num_pos + num_neg]),
                                           tf.constant([num_proposal], dtype=tf.int64))
        # selected_sample.append(idx_per_image_mask)
        # selected_sample_mask = tf.stack(selected_sample, axis=0)
        # return selected_sample_mask
        return idx_per_image_mask

    @staticmethod
    def cal_reg_target(gt_box, proposal, weights=(1.0, 1.0, 1.0, 1.0)):
        """
        :param gt_box: shape [N, 4] or [batch_size, N ,4]
        :param proposal: shape [N, 4] or [batch_size, N ,4]
        :param weights:
        :return:
        """
        wx, wy, ww, wh = weights[0], weights[1], weights[2], weights[3]

        proposal_x1, proposal_y1, proposal_x2, proposal_y2 = tf.split(proposal, [1, 1, 1, 1], axis=-1)

        gt_box_x1, gt_box_y1, gt_box_x2, gt_box_y2 = tf.split(gt_box, [1, 1, 1, 1], axis=-1)

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

        targets = tf.concat([targets_dx, targets_dy, targets_dw, targets_dh], axis=-1)
        return targets

    def label_anchors(self, gt_box):
        batch_size = gt_box.get_shape().as_list()[0]
        sampled_gt_regs, sampled_gt_labels, sampled_idxes = list(), list(), list()
        for i in range(batch_size):
            sampled_gt_reg, sampled_gt_label, sampled_idx = self._label_anchors(gt_box[i])
            sampled_gt_regs.append(sampled_gt_reg)
            sampled_gt_labels.append(sampled_gt_label)
            sampled_idxes.append(sampled_idx)

        sampled_gt_regs = tf.stack(sampled_gt_regs, axis=0)
        sampled_gt_labels = tf.stack(sampled_gt_labels, axis=0)
        sampled_idxes = tf.stack(sampled_idxes, axis=0)
        return sampled_gt_regs, sampled_gt_labels, sampled_idxes

    def _label_anchors(self, gt_box):
        matched_idxs, labels = self.encode_flag_and_match_box(gt_box)
        # print('labels.shape', labels.shape, matched_idxs.shape)
        intermediate_sample_masks = self.balanced_sample(labels)
        # print('intermediate_sample_masks shape', intermediate_sample_masks.shape)
        _, indices = tf.nn.top_k(intermediate_sample_masks, self.sample_num)
        # print('indices.shape', indices.shape)
        selected_sample_idx = tf.gather(matched_idxs, indices, axis=0, batch_dims=-1)
        # print('selected_sample_idx.shape', selected_sample_idx.shape)
        sampled_anchor = tf.gather(self.anchor, selected_sample_idx, axis=0, batch_dims=-1)
        # print('sampled_anchor.shape', sampled_anchor.shape)
        sampled_gt_box = tf.gather(gt_box, selected_sample_idx, axis=0, batch_dims=-1)
        # print('sampled_gt_box.shape', sampled_gt_box.shape)
        sampled_gt_label = tf.gather(labels, selected_sample_idx, axis=0, batch_dims=-1)
        # print('sampled_gt_label.shape', sampled_gt_label.shape)

        sampled_gt_reg = self.cal_reg_target(sampled_gt_box, sampled_anchor)
        # print('sampled_gt_reg.shape', sampled_gt_reg.shape)
        return sampled_gt_reg, sampled_gt_label, selected_sample_idx


def generate_anchor():
    for level in range(4, 5 + 1):
        boxes_l = []
        for scale in range(1):
            for aspect_ratio in [0.5, 1, 2.]:
                stride = 2 ** level
                intermediate_scale = 2 ** (scale / float(1))
                base_anchor_size = 4 * stride * intermediate_scale
                aspect_x = aspect_ratio ** -0.5
                aspect_y = aspect_ratio ** 0.5
                half_anchor_size_x = base_anchor_size * aspect_x / 2.0
                half_anchor_size_y = base_anchor_size * aspect_y / 2.0
                x = tf.range(0, 448 - stride / 2, stride)
                y = tf.range(0, 448 - stride / 2, stride)
                xv, yv = tf.meshgrid(x, y)
                xv = tf.cast(tf.reshape(xv, [-1]), dtype=tf.float32)
                yv = tf.cast(tf.reshape(yv, [-1]), dtype=tf.float32)
                # Tensor shape Nx4.
                boxes = tf.stack([
                    xv - half_anchor_size_x, yv - half_anchor_size_y,
                    xv + half_anchor_size_x, yv + half_anchor_size_y
                ],
                    axis=1)
                boxes_l.append(boxes)
        boxes_l = tf.stack(boxes_l, axis=1)
        boxes_l = tf.reshape(boxes_l, [-1, 4])
        print(boxes_l)


if __name__ == '__main__':
    pass
    anchor = Anchor(min_level=3, max_level=5, num_scales=1, aspect_ratios=(0.5, 1., 2.),
                    anchor_size=4, image_size=(448, 448))
    print(anchor.multilevel_boxes)
    # anchor = Anchor(min_level=5, max_level=5, num_scales=1,
    #                 aspect_ratios=(0.5, 1., 2.), anchor_size=4, image_size=(448, 448))
    # level_boxes = anchor.multilevel_boxes
    # anc = []
    # for i in level_boxes.keys():
    #     anc.append(tf.reshape(level_boxes[i], [-1, 4]))
    # out = tf.concat(anc, axis=-2)
    # print(out.shape)
    # print(out)
    # gt = tf.random.uniform([1, 10, 4])
    # al = AnchorLabeler(out)
    # al.label_anchors(gt)
