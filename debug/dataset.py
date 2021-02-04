import os

import h5py
import nibabel as nib
import numpy as np
from tqdm import tqdm

np.set_printoptions(threshold=224 * 224, linewidth=224)


def cal_lt_rb(matrix):
    matrix = matrix > 0.
    y, x = np.nonzero(matrix)
    y1 = np.min(y)
    x1 = np.min(x)
    y2 = np.max(y)
    x2 = np.max(x)
    return x1, y1, x2, y2, matrix.astype(np.float64)


def generate_json(root_dir='/home/wangxu/code/maskrcnn/brats17/HGG'):
    """annotation json file"""
    img = list()
    mask = list()
    cls = list()
    box = list()
    j = 0
    for file_name in tqdm(os.listdir(root_dir)):
        if file_name == '.DS_Store':
            continue
        mri_image = nib.load('{}/{}/{}_t2.nii.gz'.format(root_dir, file_name, file_name))
        mri_mask = nib.load('{}/{}/{}_seg.nii.gz'.format(root_dir, file_name, file_name))

        images = mri_image.get_fdata()
        masks = mri_mask.get_fdata()
        for i in range(masks.shape[-1]):
            cropped_image = images[8:-8, 8:-8, i]
            cropped_mask = masks[8:-8, 8:-8, i]

            if cropped_mask.max() == 0.:
                continue
            x1, y1, x2, y2, cropped_mask = cal_lt_rb(cropped_mask)
            if (x2 - x1) * (y2 - y1) == 0:
                continue
            j += 1
            cropped_image = cropped_image / cropped_image.max()

            mask.append(cropped_mask)
            box.append([[x1, y1, x2, y2]])
            img.append(cropped_image)
            cls.append([[1]])

    with h5py.File('brats17.hdf5', 'w') as h5_file:
        brats = h5_file.create_group('brats17')
        brats_image = brats.create_dataset('image', data=np.array(img))
        brats_mask = brats.create_dataset('mask', data=np.array(mask))
        brats_box = brats.create_dataset('box', data=np.array(box))
        brats_cls = brats.create_dataset('cls', data=np.array(cls), )


class Dataset:
    def __init__(self, file_path):
        self.h5_file = h5py.File(file_path, 'r')

    def __len__(self):
        return len(self.h5_file['brats17']['cls'])

    def close(self):
        self.h5_file.close()

    def load(self, idx):
        image = self.h5_file['brats17']['image'][idx]
        mask = self.h5_file['brats17']['mask'][idx]
        box = self.h5_file['brats17']['box'][idx]
        cls = self.h5_file['brats17']['cls'][idx]

        return image, mask, box, cls


if __name__ == '__main__':
    generate_json()
    # print(len(Dataset('./brats17.hdf5')))
    pass
