# -*- coding: utf-8 -*-
# @File    :   keras_data.py
# @Time    :   2023/5/8
# @Author  :   SAM
# @Desc    :


import os

from utils import ProjectPath


class KerasDataPath(object):

    def __init__(self):
        self.fashion_path = self._subfile_path('fashion')
        self._imdb_path = self._subfile_path('imdb')
        self._keras_mnist_data_path = self._subfile_path('keras_mnist_data')
        self._auto_mpg_path = self._subfile_path('auto_mpg')
        self._higgs_path = self._subfile_path('HIGGS')

    def _subfile_path(self, path):
        """
        拼接数据集路径
        :param path: 数据集类型
        :return:
        """
        return os.path.join(os.path.join(ProjectPath, 'keras_data'), path)

    def _join_path(self, dir, filename):
        return os.path.join(dir, filename)

    @property
    def imdb_data_path(self):
        return self._join_path(self._imdb_path, 'imdb.npz')

    @property
    def imdb_word_index_path(self):
        "get_word_index"
        return self._join_path(self._imdb_path, 'reuters_word_index.json')

    @property
    def helloworld_keras_mnist_data_path(self):
        _path = self._join_path(self._keras_mnist_data_path, 'datasets')
        return self._join_path(_path, 'mnist.npz')

    @property
    def auto_mpg_data_path(self):
        return self._join_path(self._auto_mpg_path, 'auto-mpg.data')

    @property
    def higgs_path(self):
        return self._join_path(self._higgs_path, 'HIGGS.csv.gz')


keras_data_path = KerasDataPath()
