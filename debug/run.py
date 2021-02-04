import math
import tensorflow as tf

try:
    from debug.backbone import resnet, fpn
    from debug.rpn import rpn, generate_proposal
    from debug.roi import select_train_sample, cal_reg_target
    from debug.spatial_transform_ops import multilevel_crop_and_resize
    from debug.head import shared_neck, predictor
    # from debug.keras_utils import maybe_enter_backend_graph
    from debug.anchor import Anchor
    from debug.dummy_generator import dummy_generator
    from debug.dataloader import data_generator
    from debug.loss import RpnBoxLoss, RpnScoreLoss, HeadBoxLoss, HeadClassLoss
except ModuleNotFoundError:
    from backbone import resnet, fpn
    from rpn import rpn, generate_proposal
    from roi import select_train_sample, cal_reg_target
    from spatial_transform_ops import multilevel_crop_and_resize
    from head import shared_neck, predictor
    # from debug.keras_utils import maybe_enter_backend_graph
    from anchor import Anchor
    from dataloader import data_generator
    from loss import RpnBoxLoss, RpnScoreLoss, HeadBoxLoss, HeadClassLoss

class HyperParams:
    image_size = (224, 224)
    min_level = 3
    max_level = 5
    aspect_ratios = (0.5, 1., 2.)
    num_anchor_per_location = 3
    anchor_size = 4

    pre_nms_top_k = 2000
    post_nms_top_k = 1000
    proposal_iou_thresh = 0.7
    roi_low_threshold = 0.3
    roi_high_threshold = 0.7

    fpn_channel = 256
    batch_size = 1

    rpn_sample_size_per_im = 256
    max_gt_num = 1
    epoch = 100


def merge_rpn_output(output, batch_size=1, box=True):
    reg_output = list()
    for reg in output:
        shape = [batch_size, -1, 4] if box else [batch_size, -1, ]
        reg_output.append(tf.reshape(reg, shape))
    reg_output = tf.concat(reg_output, axis=1)
    return reg_output


def maskrcnn(param):
    inputs = tf.keras.layers.Input(shape=param.image_size + (3,), name='image', batch_size=param.batch_size)
    input_gt_box = tf.keras.layers.Input(shape=(param.max_gt_num, 4), name='gt_box', batch_size=param.batch_size)
    input_gt_cls = tf.keras.layers.Input(shape=(param.max_gt_num, 1), name='gt_cls', batch_size=param.batch_size)
    outputs = dict()

    anchor = Anchor(min_level=param.min_level, max_level=param.max_level, num_scales=1,
                    aspect_ratios=param.aspect_ratios, anchor_size=param.anchor_size, image_size=param.image_size)

    # with maybe_enter_backend_graph():
    y3, y4, y5 = resnet(inputs)
    p3, p4, p5 = fpn(y3, y4, y5, param.fpn_channel)
    cls_rpn, reg_rpn = rpn([p3, p4, p5], param.fpn_channel, param.num_anchor_per_location)

    cls_rpn_merged = merge_rpn_output(cls_rpn, batch_size=param.batch_size, box=False)
    reg_rpn_merged = merge_rpn_output(reg_rpn, batch_size=param.batch_size, box=True)
    outputs.update({'rpn_score_output': cls_rpn_merged, 'rpn_reg_output': reg_rpn_merged})

    selected_boxes, _ = generate_proposal(cls_rpn, reg_rpn, anchor.multilevel_boxes,
                                          param.pre_nms_top_k, param.post_nms_top_k,
                                          param.image_size, param.proposal_iou_thresh)
    selected_boxes = tf.stop_gradient(selected_boxes)
    sampled_proposal, sampled_gt_box, sampled_gt_label, sampled_gt_idx = select_train_sample(input_gt_box,
                                                                                             selected_boxes,
                                                                                             input_gt_cls)

    head_reg_target = cal_reg_target(sampled_gt_box, sampled_proposal)
    outputs.update({'head_reg_target': head_reg_target, 'head_cls_target': sampled_gt_label})

    fpn_features = {3: p3, 4: p4, 5: p5}
    roi_features = multilevel_crop_and_resize(fpn_features, sampled_proposal, output_size=7)

    # with maybe_enter_backend_graph():
    shared_feature = shared_neck(roi_features)
    head_cls_output, head_reg_output = predictor(shared_feature)

    outputs.update({'head_reg_output': head_reg_output, 'head_cls_output': head_cls_output})

    model = tf.keras.Model(inputs=[inputs, input_gt_cls, input_gt_box],
                           outputs=outputs)
    return model


def create_loss_fn(param):
    def total_loss_fn(labels, outputs):
        rpn_score_loss_fn = RpnScoreLoss(param.rpn_sample_size_per_im)
        rpn_box_loss_fn = RpnBoxLoss()

        rpn_score_loss = rpn_score_loss_fn(outputs['rpn_score_output'], labels['rpn_score_target'])
        rpn_box_loss = rpn_box_loss_fn(outputs['rpn_reg_output'], labels['rpn_reg_target'])

        head_class_loss_fn = HeadClassLoss()
        head_box_loss_fn = HeadBoxLoss()
        head_class_loss = head_class_loss_fn(outputs['head_cls_output'], outputs['head_cls_target'])
        head_box_loss = head_box_loss_fn(outputs['head_reg_output'],
                                         outputs['head_cls_target'],
                                         outputs['head_reg_target'])

        return {'head_cls_loss': head_class_loss,
                'head_reg_loss': head_box_loss,
                'rpn_score_loss': rpn_score_loss,
                'rpn_reg_loss': rpn_box_loss}

    return total_loss_fn


def create_train_step(model, loss_fn, optimizer):
    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            output = model(x, training=True)
            loss_value = loss_fn(y, output)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        # train_acc_metric.update_state(y, logits)
        return loss_value

    return train_step


def create_data(param):
    hw = param.image_size[0] * param.image_size[1]
    anchor_num = sum(list(map(lambda x: hw / math.pow(2, x) ** 2,
                              list(range(param.min_level, param.max_level + 1)))))
    anchor_num = int(anchor_num) * param.num_anchor_per_location
    output_signature = {'image': tf.TensorSpec(shape=(param.batch_size,) + param.image_size + (3,), dtype=tf.float32),
                        'gt_box': tf.TensorSpec(shape=(param.batch_size,) + (param.max_gt_num,) + (4,),
                                                dtype=tf.float32),
                        'gt_cls': tf.TensorSpec(shape=(param.batch_size,) + (param.max_gt_num,) + (1,),
                                                dtype=tf.float32),
                        'rpn_reg': tf.TensorSpec(shape=(param.batch_size,) + (anchor_num,) + (4,), dtype=tf.float32),
                        'rpn_cls': tf.TensorSpec(shape=(param.batch_size,) + (anchor_num,), dtype=tf.float32),
                        'anchor': tf.TensorSpec(shape=(anchor_num,) + (4,), dtype=tf.float32), }

    train_data = tf.data.Dataset.from_generator(data_generator, output_signature=output_signature)
    return train_data


def main():
    param = HyperParams

    net = maskrcnn(param)
    net_loss_fn = create_loss_fn(param)

    net_optimizer = tf.keras.optimizers.Adam()
    train_data = create_data(param)
    one_step = create_train_step(net, net_loss_fn, net_optimizer)

    train(one_step, train_data, param)


def train(one_step, train_data, param):
    for epoch in range(param.epoch):
        for step, batch_data in enumerate(train_data):
            label = {'rpn_score_target': batch_data['rpn_cls'], 'rpn_reg_target': batch_data['rpn_reg']}
            loss = one_step([batch_data['image'], batch_data['gt_cls'], batch_data['gt_box']], label)
            print('head_cls_loss {}  head_reg_loss {}  rpn_score_loss {}  rpn_reg_loss {}'.format(
                loss['head_cls_loss'].numpy(), loss['head_reg_loss'].numpy(),
                loss['rpn_score_loss'].numpy(), loss['rpn_reg_loss'].numpy()))


if __name__ == '__main__':
    main()
