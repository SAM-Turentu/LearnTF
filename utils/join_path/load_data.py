# ！/usr/bin/python3
# -*- coding: utf-8 -*-
# @TIme 2023/6/7 17:33
# @Author: SAM
# @File: load_data.py
# @Email: SAM-Turentu@outlook.com
# @Desc:


import os

from utils import ProjectPath


class LoadDataPath(object):

    def __init__(self):
        self._load_data = os.path.join(ProjectPath, 'load_data')

    def _join_path(self, dir, filename):
        return os.path.join(dir, filename)

    @property
    def images_path(self):
        """直接使用解压后的文件夹"""
        return self._join_path(self._load_data, 'flower_photos')


load_data = LoadDataPath()
