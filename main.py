import numpy as np
import tensorflow as tf
from tensorflow import keras

from network.rpn import rpn, generate_proposal
from network.backbone import resnet, fpn
from utils.roi import select_train_sample
from loss.rpn import RPNClsLoss, RPNRegLoss, RPNLoss
from data.data_loader import data_generator, dummy_generator

# tf.config.gpu.set_per_process_memory_fraction(0.75)
# tf.config.gpu.set_per_process_memory_growth(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


def get_fake_dataset():
    def map_f(img, rpn_cls_gt, rpn_reg_gt, rpn_pos_sample_idx, rpn_neg_sample_idx):
        inputs = {"image": img,
                  "rpn_cls_gt": rpn_cls_gt,
                  "rpn_reg_gt": rpn_reg_gt,
                  "rpn_pos_sample_idx": rpn_pos_sample_idx,
                  "rpn_sample_idx": rpn_neg_sample_idx}
        targets = {}
        return inputs, targets

    fake_imgs = np.ones([10, 448, 448, 3])
    fake_cls = np.ones([10, 12495, 1])
    fake_reg = np.ones([10, 12495, 4])
    fake_pos_idx = np.ones([10, 12495, 1])
    fake_idx = np.ones([10, 12495, 1])

    fake_dataset = tf.data.Dataset.from_tensor_slices(
        (fake_imgs, fake_cls, fake_reg, fake_pos_idx, fake_idx)
    ).map(map_f).batch(1)
    return fake_dataset


def maskrcnn():
    inputs = keras.layers.Input(shape=(448, 448, 3), name='image')
    # input_rpn_cls = keras.layers.Input([12495, 1], name='rpn_cls_gt')
    # input_rpn_reg = keras.layers.Input([12495, 4], name='rpn_reg_gt')
    # input_rpn_pos_sample_idx = keras.layers.Input([12495, 1], name='rpn_pos_sample_idx')
    # input_rpn_sample_idx = keras.layers.Input([12495, 1], name='rpn_sample_idx')
    input_grid_anchor = keras.layers.Input([12495, 4], name='grid_anchor')
    input_roi_gt_reg = keras.layers.Input([2000, 4], name='roi_gt_reg')
    input_roi_gt_cls = keras.layers.Input([2000, 1], name='roi_gt_cls')

    y3, y4, y5 = resnet(inputs)
    p3, p4, p5, p6 = fpn(y3, y4, y5)
    cls, reg = rpn([p3, p4, p5, p6], 256, 3)
    # rpn_cls_loss = RPNClsLoss()([input_rpn_cls, cls, input_rpn_sample_idx])
    # rpn_reg_loss = RPNRegLoss()([input_rpn_reg, reg, input_rpn_pos_sample_idx])

    # cls_loss, reg_loss = RPNLoss()(
    #     [input_rpn_cls, cls, input_rpn_reg, reg, input_rpn_sample_idx, input_rpn_pos_sample_idx])
    proposal_box = generate_proposal(cls, reg, input_grid_anchor,
                                     200, [3 * 56 * 56, 3 * 28 * 28, 3 * 14 * 14, 3 * 7 * 7],
                                     (448, 448), 0.01, 0.7)
    # print(proposal_box)
    matched_idxs, flags = select_train_sample(input_roi_gt_reg, proposal_box, input_roi_gt_cls)
    # model = keras.Model(inputs=[inputs, input_rpn_cls, input_rpn_reg,
    #                             input_rpn_pos_sample_idx, input_rpn_sample_idx,
    #                             input_grid_anchor],
    #                     outputs=[cls, reg, cls_loss, reg_loss, proposal_box])
    model = keras.Model(inputs=[inputs, input_grid_anchor, input_roi_gt_reg, input_roi_gt_cls],
                        outputs=[matched_idxs, flags])
    return model


m = maskrcnn()

m.summary(line_length=170)
# for i in range(len(m.layers)):
#     print(m.get_layer(index=i).output)
# keras.utils.plot_model(m, to_file='model.png')

# dummy_data = tf.data.Dataset.from_generator(dummy_generator,
#                                             (tf.float64, tf.float64, tf.float64, tf.float64, tf.float64))
# dataset = dummy_data.map(map_fn)
# dataset = dummy_data.batch(batch_size=10)
# dataset = dummy_data.repeat(count=2)
# for batch, (x, y, z, c, v) in enumerate(dataset):
#     print(batch)
# m.compile(optimizer="adam")
# m.fit(dummy_generator(), epochs=2, steps_per_epoch=10)
