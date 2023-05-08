# -*- coding: utf-8 -*-
# @File    :   hub_model_join_path.py
# @Time    :   2023/5/8
# @Author  :   SAM
# @Desc    :


import os

from utils import ProjectPath


class HubModelPath(object):

    def __init__(self):
        self.hub_models = os.path.join(ProjectPath, 'hub_models')

    def model_path(self, file):
        return os.path.join(self.hub_models, file)


hub_model_path = HubModelPath()

