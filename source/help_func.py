import os
import random

import yaml
import torch
import numpy as np


def read_yaml(path_list):
    cfg = dict()
    for path in path_list:
        print(f'Loading Configure File: {path}')
        with open(path, 'r', encoding='utf-8') as file:
            base_cfg = yaml.safe_load(file)
            cfg.update(base_cfg)
    return cfg


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def init_folder(path):
    make_dir(path)
    make_dir(os.path.join(path, 'Logs'))
    make_dir(os.path.join(path, 'Weights'))


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def tensor2array(tensor, squeeze=False):
    if squeeze:
        tensor = tensor.squeeze()
    return tensor.detach().cpu().numpy()


class Recorder(object):
    """
    record the metric and return the statistic results
    """
    def __init__(self):
        self.data = dict()
        self.keys = []

    def update(self, item):
        for key in item.keys():
            if key not in self.keys:
                self.keys.append(key)
                self.data[key] = []

            self.data[key].append(item[key])

    def reset(self, keys=None):
        if keys is None:
            keys = self.data.keys()
        for key in keys:
            self.data[key] = []

    def call(self, key, return_std=False):
        arr = np.array(self.data[key])
        if return_std:
            return np.mean(arr), np.std(arr)
        else:
            return np.mean(arr)

    def average(self):
        average_dict = {}
        for key in self.keys:
            average_dict[key] = np.asarray(self.data[key]).flatten().mean()

        return average_dict

    def stddev(self):
        stddev_dict = {}
        for key in self.keys:
            stddev_dict[key] = np.std(np.concatenate(self.data[key]))

        return stddev_dict


