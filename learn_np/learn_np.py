# -*- coding: utf-8 -*-
# @File    :   learn_np.py
# @Time    :   2023/5/4
# @Author  :   SAM
# @Desc    :   学习 numpy


import tensorflow as tf

from utils.utils import tools


def main():
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = tools.load_fashion_data()  # fashion_mnist.load_data()
    # train_images, test_images = train_images / 255, test_images / 255

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    class_names_ch = ['T恤/上衣', '裤子', '套头衫', '连衣裙', '外套', '凉鞋', '衬衫', '运动鞋', '包', '短靴']


if __name__ == '__main__':
    main()
