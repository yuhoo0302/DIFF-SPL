import os
import random

import numpy as np
import nibabel as nib

from plane_func import Slicer


def read_ids(path):
    id_list = open(path, 'r').readlines()
    id_list = [idx.rstrip() for idx in id_list]

    return id_list


def load_list(data_path, text_path):
    def __get_dict(path, subject_id):
        __subj_path = os.path.join(path, subject_id)
        return {
            "Volume": os.path.join(__subj_path, "volume.nii.gz"),
            "Tangent": os.path.join(__subj_path, "tangent.npy"),
            "ID": subject_id,
        }

    train_list = []
    val_list = []
    test_list = []

    train_subjects = read_ids(os.path.join(text_path, "train.txt"))
    val_subjects = read_ids(os.path.join(text_path, "val.txt"))
    test_subjects = read_ids(os.path.join(text_path, "test.txt"))

    for subject in train_subjects:
        subject_dict = __get_dict(data_path, subject)
        train_list.append(subject_dict)

    for subject in val_subjects:
        subject_dict = __get_dict(data_path, subject)
        val_list.append(subject_dict)

    for subject in test_subjects:
        subject_dict = __get_dict(data_path, subject)
        test_list.append(subject_dict)

    print("Train/val/test with", "{}/{}/{}".format(len(train_list), len(val_list), len(test_list)), "subjects")

    return train_list, val_list, test_list


class SPLDataSet(object):
    def __init__(self, data_list, batch_size, plane_size):
        self.data_list = data_list
        self.batch_size = batch_size
        self.plane_size = plane_size
        self.data_num = self.data_list

    def pop_data(self):
        data_info = random.choice(self.data_list)
        volume = nib.load(data_info['Volume']).get_fdata()
        volume = volume/255.
        tangent = np.load(data_info['Tangent'])

        return volume, tangent

    def pop_data_idx(self, idx):
        data_info = self.data_list[idx]
        volume = nib.load(data_info['Volume']).get_fdata()
        volume = volume/255.
        tangent = np.load(data_info['Tangent'])
        name = data_info['Volume'].split('/')[-2]

        return volume, tangent, name

    @property
    def num(self):
        length = len(self.data_num)
        return length


def load_dataset(cfg):

    train_list, val_list, test_list = load_list(cfg['DataPath'], cfg['DataSplitPath'])

    dataset_dict = dict()

    for mode, subj_list in zip(['Train', 'Val', 'Test'], [train_list, val_list, test_list]):

        if cfg['DataSetType'] == 'UterusC' or cfg['DataSetType'] == 'FetalTC':
            dataset = SPLDataSet(data_list=subj_list,
                                 batch_size=cfg['BatchSize'],
                                 plane_size=cfg['PlaneSize'])
        else:
            raise NotImplementedError

        dataset_dict[mode] = dataset

    return dataset_dict

