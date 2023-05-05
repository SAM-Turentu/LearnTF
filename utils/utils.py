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

from utils import ProjectPath


class Utils(object):

    def __init__(self):
        self.fashion_path = self._mnistdata_path('fashion')
        self.mnist_path = self._mnistdata_path('mnist')

    def _mnistdata_path(self, path):
        """
        拼接数据集路径
        :param path: 数据集类型
        :return:
        """
        return os.path.join(os.path.join(ProjectPath, 'mnist_data'), path)

    def load_fashion_data(self):
        files = [
            "train-labels-idx1-ubyte.gz",
            "train-images-idx3-ubyte.gz",
            "t10k-labels-idx1-ubyte.gz",
            "t10k-images-idx3-ubyte.gz",
        ]
        paths = []
        for file in files:
            paths.append(os.path.join(self.fashion_path, file))

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


tools = Utils()

__all__: 'tools'
