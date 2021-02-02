import numpy as np
import tensorflow as tf
from debug.backbone import resnet, fpn
from debug.rpn import rpn, generate_proposal
from debug.roi import select_train_sample, cal_reg_target
from debug.spatial_transform_ops import multilevel_crop_and_resize
from debug.head import shared_neck, predictor
from debug.keras_utils import maybe_enter_backend_graph


def maskrcnn():
    inputs = tf.keras.layers.Input(shape=(448, 448, 3), name='image', batch_size=2)
    # input_grid_anchor = tf.keras.layers.Input([12495, 4], name='grid_anchor', batch_size=2)
    input_roi_gt_reg = tf.keras.layers.Input([2000, 4], name='roi_gt_reg', batch_size=2)
    input_roi_gt_cls = tf.keras.layers.Input([2000, 1], name='roi_gt_cls', batch_size=2)

    # image = tf.constant(np.random.rand(2, 448, 448, 3).astype(np.float32))
    anchor = tf.constant(np.random.rand(12495, 4).astype(np.float32))
    # roi_gt_reg = tf.constant(np.random.rand(2, 10, 4).astype(np.float32))
    # roi_gt_cls = tf.constant(np.random.rand(2, 10, 1).astype(np.float32))
    outputs = dict()
    with maybe_enter_backend_graph():
        y3, y4, y5 = resnet(inputs)
        p3, p4, p5, p6 = fpn(y3, y4, y5)
        cls_rpn, reg_rpn = rpn([p3, p4, p5, p6], 256, 3)

    for i in range(len(cls_rpn)):
        outputs.update({'rpn_cls_output_level{}'.format(i + 3): cls_rpn[i],
                        'rpn_reg_output_level{}'.format(i + 3): reg_rpn[i]})

    selected_boxes, _ = generate_proposal(cls_rpn, reg_rpn, anchor, 2000, 1000, (448, 448), 0.7, 0.01)
    selected_boxes = tf.stop_gradient(selected_boxes)
    sampled_proposal, sampled_gt_box, sampled_gt_label, sampled_gt_idx = select_train_sample(input_roi_gt_reg,
                                                                                             selected_boxes,
                                                                                             input_roi_gt_cls)

    head_reg_target = cal_reg_target(sampled_gt_box, sampled_proposal)
    outputs.update({'head_reg_target': head_reg_target, 'head_cls_target': sampled_gt_label})
    fpn_features = {3: p3, 4: p4, 5: p5, 6: p6}
    roi_features = multilevel_crop_and_resize(fpn_features, sampled_proposal, output_size=7)

    with maybe_enter_backend_graph():
        shared_feature = shared_neck(roi_features)
        head_cls_output, head_reg_output = predictor(shared_feature)

    outputs.update({'head_reg_output': head_reg_output, 'head_cls_output': head_cls_output})

    # print(sampled_proposal.shape)
    # print(sampled_gt_box.shape)
    # print(sampled_gt_label.shape)
    # print(sampled_gt_idx.shape)
    # print(roi_features.shape)
    # print(shared_feature.shape)
    # print(head_reg_output.shape)
    # print(head_cls_output.shape)

    model = tf.keras.Model(inputs=[inputs, input_roi_gt_cls, input_roi_gt_reg],
                           outputs=outputs)
    return model


m = maskrcnn()
m.summary()
# image = tf.constant(np.random.rand(2, 448, 448, 3).astype(np.float32))
# for i in range(2):
#     print(image[i].shape)
