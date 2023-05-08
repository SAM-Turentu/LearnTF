# -*- coding: utf-8 -*-
# @File    :   mnist_data_join_path.py
# @Time    :   2023/5/8
# @Author  :   SAM
# @Desc    :


import os

from utils import ProjectPath


class MnistDataPath(object):

    def __init__(self):
        self.fashion_path = self._mnistdata_path('fashion')
        self.mnist_path = self._mnistdata_path('mnist')
        self.imdb_path = self._mnistdata_path('imdb')

    def _mnistdata_path(self, path):
        """
        拼接数据集路径
        :param path: 数据集类型
        :return:
        """
        return os.path.join(os.path.join(ProjectPath, 'mnist_data'), path)

    def _tf_datasets_path(self, path):
        return os.path.join(os.path.join(ProjectPath, 'tf_datasets'), path)

    def join_path(self, dir, filename):
        return os.path.join(dir, filename)

    @property
    def imdb_data_path(self):
        return self.join_path(self.imdb_path, 'imdb.npz')

    @property
    def imdb_word_index_path(self):
        "get_word_index"
        return self.join_path(self.imdb_path, 'reuters_word_index.json')


mnist_data_path = MnistDataPath()