import tensorflow as tf

from core.anchor import Anchor
from core.spatial_transform_ops import multilevel_crop_and_resize
from network.backbone import resnet
from network.fpn import fpn
from network.rpn import rpn, multilevel_propose_rois
from network.roi import assign_and_sample_proposals
import keras_utils
from utils import box_utils

with keras_utils.maybe_enter_backend_graph():
    batch_size = 2
    input_layer = {
        'image': tf.keras.layers.Input(shape=(1024, 1024, 3), batch_size=batch_size, name='image', dtype=tf.float32),
        'image_info': tf.keras.layers.Input(shape=[4, 2], batch_size=batch_size, name='image_info'),
        'gt_boxes': tf.keras.layers.Input(shape=[100, 4], batch_size=batch_size, name='gt_boxes'),
        'gt_classes': tf.keras.layers.Input(shape=[100], batch_size=batch_size, name='gt_classes', dtype=tf.int64)}

    model_outputs = {}

    image = input_layer['image']
    # _, image_height, image_width, _ = image.get_shape().as_list()

    backbone_features = resnet(image)
    fpn_features = fpn(backbone_features)
    rpn_features = rpn(fpn_features, 256, 3)
    input_anchor = Anchor(3, 6, 1, [1.0, 2.0, 0.5], 8, (1024, 1024))
    selected_rois, selected_roi_scores = multilevel_propose_rois(rpn_features[1], rpn_features[0],
                                                                 input_anchor.multilevel_boxes,
                                                                 input_layer['image_info'][:, 1, :],
                                                                 use_batched_nms=False)
    selected_rois = tf.stop_gradient(selected_rois)

    rpn_rois, matched_gt_boxes, matched_gt_classes, matched_gt_indices = (
        assign_and_sample_proposals(selected_rois, input_layer['gt_boxes'], input_layer['gt_classes']))

    box_targets = box_utils.encode_boxes(matched_gt_boxes, rpn_rois, weights=[10.0, 10.0, 5.0, 5.0])
    # If the target is background, the box target is set to all 0s.
    box_targets = tf.where(tf.tile(tf.expand_dims(tf.equal(matched_gt_classes, 0), axis=-1),
                                   [1, 1, 4]), tf.zeros_like(box_targets), box_targets)
    roi_features = multilevel_crop_and_resize(fpn_features, rpn_rois, output_size=7)

    print(roi_features)
# model_outputs.update({'rpn_score_outputs': tf.nest.map_structure(lambda x: tf.cast(x, tf.float32),
#                                                                  rpn_features[0]),
#                       'rpn_box_outputs': tf.nest.map_structure(lambda x: tf.cast(x, tf.float32),
#                                                                rpn_features[1])})
# m = tf.keras.Model(inputs=input_layer, outputs=model_outputs)
# m.summary()
