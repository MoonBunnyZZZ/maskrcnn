import nibabel as nib
import numpy as np
import sys


def tf_data():
    import tensorflow as tf
    dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
    # print(type(dataset))
    tf.data.Dataset.from_generator()


def nib():
    np.set_printoptions(threshold=sys.maxsize)
    img = nib.load('D:\\data\\brats17\\HGG\\Brats17_2013_2_1\\Brats17_2013_2_1_seg.nii.gz')
    mask = img.get_fdata()

    nii_data = nib.load('D:\\data\\brats17\\HGG\\Brats17_2013_2_1\\Brats17_2013_2_1_t1.nii.gz')
    img = nii_data.dataobj
    # print(img[::, 100].sum())
    for i in range(img.shape[-1]):
        print(mask[::, i].max())


a = np.array([[0, 1], [2, 0]])
# a = a > 0
print(a)
print(a.max() == 2)
print(np.argmax(a, axis=0))
print(np.argmax(a, axis=1))
