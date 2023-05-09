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


tf_datasets = TFDatasetsPath()