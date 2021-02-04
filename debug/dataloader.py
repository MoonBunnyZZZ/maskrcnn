import numpy as np
from debug.dummy_generator import set_cell_anchors, set_grid_anchors, cal_bbox_from_mask, rpn_target
from debug.dataset import Dataset


def data_generator():
    brats = Dataset('./brats17.hdf5')
    cell_anchors = set_cell_anchors(((32,), (64,), (128,)), ((0.5, 1, 2),) * 3)
    anchors = set_grid_anchors([[28, 28], [14, 14], [7, 7]], [[8, 8], [16, 16], [32, 32]], cell_anchors)

    item_idx = 0
    batch_image, batch_gt_box, batch_gt_cls, batch_rpn_reg, batch_rpn_cls = list(), list(), list(), list(), list()

    while True:
        image, gt_mask, gt_box, gt_cls = brats.load(item_idx)
        # image, gt_mask = augment_data(image, gt_mask)

        rpn_cls, rpn_reg = rpn_target(anchors, gt_box, 0.5, 256)

        batch_image.append(np.repeat(np.reshape(image, (224, 224, 1)), 3, axis=-1))
        batch_gt_cls.append(gt_cls)
        batch_gt_box.append(gt_box)
        batch_rpn_reg.append(rpn_reg)
        batch_rpn_cls.append(rpn_cls)

        if (item_idx + 1) % 1 == 0:
            batch_data = {'image': np.array(batch_image),
                          'gt_box': np.array(batch_gt_box),
                          'gt_cls': np.array(batch_gt_cls),
                          'rpn_reg': np.array(batch_rpn_reg),
                          'rpn_cls': np.array(batch_rpn_cls),
                          'anchor': np.concatenate(anchors, axis=0)}
            yield batch_data
            item_idx += 1
            batch_image.clear()
            batch_gt_box.clear()
            batch_gt_cls.clear()
            batch_rpn_reg.clear()
            batch_rpn_cls.clear()
        if item_idx == len(brats):
            item_idx = 0


if __name__ == '__main__':
    gen = data_generator()

    for i, batch_data in enumerate(gen):
        # num = batch_data['gt_box'].shape[1]
        print(batch_data['image'].shape,
              batch_data['gt_box'].shape,
              batch_data['gt_cls'].shape,
              batch_data['rpn_reg'].shape,
              batch_data['rpn_cls'].shape,
              batch_data['anchor'].shape, )
        # print(i)
