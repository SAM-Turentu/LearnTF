# -*- coding: utf-8 -*-
# @File    :   utils.py
# @Time    :   2023/5/5
# @Author  :   SAM
# @Desc    :
import gzip
import os.path

import numpy as np


def switch_platform():
    """
    选择系统平台， mac, windows
    :return:
    """
    ...


def load_fishion_file():
    files = [
        "train-labels-idx1-ubyte.gz",
        "train-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
    ]
    paths = []
    for file in files:
        a = os.getcwd()
        _path = os.path.join(os.getcwd(), 'mnist_data')
        _path = os.path.join(_path, 'fashion')
        _path = os.path.join(_path, file)
        paths.append(_path)

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
