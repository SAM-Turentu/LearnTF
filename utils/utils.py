# ！/usr/bin/python3
# -*- coding: utf-8 -*-
# @TIme 2023/5/5 19:05
# @Author: SAM
# @File: utils.py
# @Email: SAM-Turentu@outlook.com
# @Desc:


import gzip
import os.path
import numpy as np
from keras import backend
from keras.datasets.cifar import load_batch

from utils import join_path


def switch_platform():
    """
    选择系统平台， mac, windows
    :return:
    """
    ...


class Utils(object):

    def __init__(self):
        ...

    def load_fashion_data(self):
        files = [
            "train-labels-idx1-ubyte.gz",
            "train-images-idx3-ubyte.gz",
            "t10k-labels-idx1-ubyte.gz",
            "t10k-images-idx3-ubyte.gz",
        ]
        paths = []
        for file in files:
            paths.append(os.path.join(join_path.keras_data_path.fashion_path, file))

        with gzip.open(paths[0], "rb") as lbpath:
            y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

        with gzip.open(paths[1], "rb") as imgpath:
            x_train = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(
                len(y_train), 28, 28
            )

        with gzip.open(paths[2], "rb") as lbpath:
            y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

        with gzip.open(paths[3], "rb") as imgpath:
            x_test = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(
                len(y_test), 28, 28
            )

        return (x_train, y_train), (x_test, y_test)

    def load_cifar10_data(self, del_path='BaseLearnTF'):
        path = join_path.keras_data_path.cifar10_path
        if del_path in path:
            path = path.replace(f'{del_path}\\', '')

        num_train_samples = 50000

        x_train = np.empty((num_train_samples, 3, 32, 32), dtype="uint8")
        y_train = np.empty((num_train_samples,), dtype="uint8")

        for i in range(1, 6):
            fpath = os.path.join(path, "data_batch_" + str(i))
            (
                x_train[(i - 1) * 10000: i * 10000, :, :, :],
                y_train[(i - 1) * 10000: i * 10000],
            ) = load_batch(fpath)

        fpath = os.path.join(path, "test_batch")
        x_test, y_test = load_batch(fpath)

        y_train = np.reshape(y_train, (len(y_train), 1))
        y_test = np.reshape(y_test, (len(y_test), 1))

        if backend.image_data_format() == "channels_last":
            x_train = x_train.transpose(0, 2, 3, 1)
            x_test = x_test.transpose(0, 2, 3, 1)

        x_test = x_test.astype(x_train.dtype)
        y_test = y_test.astype(y_train.dtype)

        return (x_train, y_train), (x_test, y_test)


tools = Utils()

__all__: [
    'tools',
]
