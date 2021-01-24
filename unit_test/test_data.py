import nibabel as nib
import numpy as np
import sys


def tf_data():
    import tensorflow as tf
    dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
    # print(type(dataset))
    tf.data.Dataset.from_generator()


def nib_read():
    np.set_printoptions(threshold=sys.maxsize)
    img = nib.load('D:\\data\\brats17\\HGG\\Brats17_2013_2_1\\Brats17_2013_2_1_seg.nii.gz')
    mask = img.get_fdata()

    nii_data = nib.load('D:\\data\\brats17\\HGG\\Brats17_2013_2_1\\Brats17_2013_2_1_t1.nii.gz')
    img = nii_data.dataobj
    # print(img[::, 100].sum())
    for i in range(img.shape[-1]):
        print(mask[::, i].max())


def np_argmax():
    a = np.array([[0, 1], [2, 0]])
    # a = a > 0
    print(a)
    print(a.max() == 2)
    print(np.argmax(a, axis=0))
    print(np.argmax(a, axis=1))


def gen():
    from data.data_loader import data_generator
    generator = data_generator(
        '/Users/wangxu_macair/code/hxk/maskrcnn/brats17.hdf5',
        ((32,), (64,), (128,), (256,), (512,)),
        ((0.5, 1, 2),) * 5,
        ((112, 112), (56, 56), (32, 32), (16, 16), (8, 8)),
        ((2, 2), (4, 4), (8, 8), (16, 16), (32, 32)), 3)
    for _ in generator:
        pass


gen()
