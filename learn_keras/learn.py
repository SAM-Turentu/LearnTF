# -*- coding: utf-8 -*-
# @File    :   learn.py
# @Time    :   2023/5/5
# @Author  :   SAM
# @Desc    :   keras 机器学习基础


# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

from utils import utils

print(tf.__version__)

fashion_mnist = tf.keras.datasets.fashion_mnist

utils.load_fishion_file()

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

