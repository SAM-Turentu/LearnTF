# -*- coding: utf-8 -*-
# @File    :   tf_datasets.py
# @Time    :   2023/5/8
# @Author  :   SAM
# @Desc    :


import os

from utils import ProjectPath


class TFDatasetsPath(object):

    def __init__(self):
        self.tf_datasets = os.path.join(ProjectPath, 'tf_datasets')

    @property
    def _titanic(self):
        return os.path.join(self.tf_datasets, 'titanic')

    def titanic_file_path(self, kind='train'):
        if kind == 'train':
            path = os.path.join(self._titanic, 'train.csv')
        else:
            path = os.path.join(self._titanic, 'eval.csv')
        return path


tf_datasets = TFDatasetsPath()
