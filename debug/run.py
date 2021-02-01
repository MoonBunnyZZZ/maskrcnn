import numpy as np
import tensorflow as tf
from debug.backbone import resnet, fpn
from debug.rpn import rpn, generate_proposal
from debug.roi import select_train_sample
from debug.spatial_transform_ops import multilevel_crop_and_resize
from debug.head import shared_neck, predictor

def maskrcnn():
    # inputs = tf.keras.layers.Input(shape=(448, 448, 3), name='image')
    # input_grid_anchor = tf.keras.layers.Input([12495, 4], name='grid_anchor')
    # input_roi_gt_reg = tf.keras.layers.Input([2000, 4], name='roi_gt_reg')
    # input_roi_gt_cls = tf.keras.layers.Input([2000, 1], name='roi_gt_cls')

    image = tf.constant(np.random.rand(2, 448, 448, 3).astype(np.float32))
    anchor = tf.constant(np.random.rand(12495, 4).astype(np.float32))
    roi_gt_reg = tf.constant(np.random.rand(2, 10, 4).astype(np.float32))
    roi_gt_cls = tf.constant(np.random.rand(2, 10, 1).astype(np.float32))

    y3, y4, y5 = resnet(image)
    p3, p4, p5, p6 = fpn(y3, y4, y5)
    cls_rpn, reg_rpn = rpn([p3, p4, p5, p6], 256, 3)
    # rpn_cls_loss = RPNClsLoss()([input_rpn_cls, cls, input_rpn_sample_idx])
    # rpn_reg_loss = RPNRegLoss()([input_rpn_reg, reg, input_rpn_pos_sample_idx])

    # cls_loss, reg_loss = RPNLoss()(
    #     [input_rpn_cls, cls, input_rpn_reg, reg, input_rpn_sample_idx, input_rpn_pos_sample_idx])
    selected_boxes, selected_scores = generate_proposal(cls_rpn, reg_rpn, anchor, 2000, 1000, (448, 448), 0.7, 0.01)

    sampled_proposal, sampled_gt_box, sampled_gt_label, sampled_gt_idx = select_train_sample(roi_gt_reg,
                                                                                             selected_boxes,
                                                                                             roi_gt_cls)

    # Create bounding box training targets.
    fpn_features = {3: p3, 4: p4, 5: p5, 6: p6}
    roi_features = multilevel_crop_and_resize(fpn_features, sampled_proposal, output_size=7)

    shared_feature = shared_neck(roi_features)
    cls_head, reg_head = predictor(shared_feature)

    print(sampled_proposal.shape)
    print(sampled_gt_box.shape)
    print(sampled_gt_label.shape)
    print(sampled_gt_idx.shape)
    print(roi_features.shape)
    print(shared_feature.shape)
    print(cls_head.shape)
    print(reg_head.shape)
    # model = keras.Model(inputs=[inputs, input_rpn_cls, input_rpn_reg,
    #                             input_rpn_pos_sample_idx, input_rpn_sample_idx,
    #                             input_grid_anchor],
    #                     outputs=[cls, reg, cls_loss, reg_loss, proposal_box])
    # model = tf.keras.Model(inputs=[inputs, input_grid_anchor],
    #                        outputs=[proposal_box])
    # return model


maskrcnn()
# image = tf.constant(np.random.rand(2, 448, 448, 3).astype(np.float32))
# for i in range(2):
#     print(image[i].shape)
