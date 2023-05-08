# -*- coding: utf-8 -*-
# @File    :   draw_result.py
# @Time    :   2023/5/6
# @Author  :   SAM
# @Desc    :   将训练结果转换为图片


import matplotlib.pyplot as plt
import numpy as np

from config.config import FishionNamesCh


class DrawResult(object):

    def plot_image(self, i, predictions_array, true_lable, img):
        """
        显示每个图像是否正确 正确蓝色，错误红色
        :param i:
        :param predictions_array:
        :param true_lable:
        :param img:
        :return:
        """
        true_lable, img = true_lable[i], img[i]
        plt.grid()
        plt.xticks([])  # 设置x轴刻度 和 标签
        plt.yticks([])

        predictions_label = np.argmax(predictions_array)
        if predictions_label == true_lable:
            color = 'blue'  # 识别正确，设置为蓝色
        else:
            color = 'red'  # 识别错误，设置为红色
        plt.xlabel(
            f"{FishionNamesCh[predictions_label]} {100 * np.max(predictions_array) :.2f%} ({FishionNamesCh[true_lable]})",
            color=color,
            fontproperties='FangSong')

    def plot_value_array(self, i, predictions_array, true_label):
        true_label = true_label[i]
        plt.grid(False)
        plt.xticks(range(10))
        plt.yticks([])
        thisplot = plt.bar(range(10), predictions_array, color='#777777')
        plt.ylim([0, 1])
        predictions_label = np.argmax(predictions_array)

        thisplot[predictions_label].set_color('red')
        thisplot[true_label].set_color('blue')
