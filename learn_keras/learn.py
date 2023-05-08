# -*- coding: utf-8 -*-
# @File    :   learn.py
# @Time    :   2023/5/5
# @Author  :   SAM
# @Desc    :   keras 机器学习基础


import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

from config.config import FishionNames
from utils.utils import tools


def main():
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = tools.load_fashion_data()  # fashion_mnist.load_data()
    # print(train_images.shape)  # (60000, 28, 28)  # 6w个训练集，每个图片都是28*28像素位点，黑白色 RGB 数值
    # print(train_labels)  # 6w个训练集标签

    # region 将 28*28 转换为图片 (matplotlib)
    # plt.figure()
    # plt.imshow(train_images[0])
    # plt.colorbar()
    # plt.grid(False)
    # plt.show()

    # plt.figure(figsize=(10, 10))
    # for i in range(25):
    #     plt.subplot(5, 5, i + 1)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.grid(False)
    #     plt.imshow(train_images[i], cmap=plt.cm.binary)
    #     plt.xlabel(str(FishionNames[train_labels[i]]), fontproperties='FangSong')  # 标签为中文必须设置字体
    # plt.show()
    # endregion

    train_images, test_images = train_images / 255, test_images / 255

    # 构建神经网络需要先配置模型的层，再编译模型
    model = tf.keras.Sequential([
        # 设置层
        tf.keras.layers.Flatten(input_shape=(28, 28)),  # 第一层：将图像格式28*28像素转换成一维数组，平展像素，格式化数据
        # Dense 密集连接 或 全连接神经层
        tf.keras.layers.Dense(128, activation='relu'),  # 第二层：128个节点（神经元）
        tf.keras.layers.Dense(10),  # 返回一个长度为 10 的 logits 数组，每个节点都包含一个得分，用来表示当前图像属于10类中的哪类
    ])

    # 编译模型
    model.compile(optimizer='adam',  # 优化器：决定模型如何根据其看到的数据和自身的损失函数进行更新
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # 损失函数：测量模型在训练期间的准确程度
                  metrics=['accuracy'])  # 指标  用于监控训练和测试步骤，  accuracy 准确率，即被正确分类的图像的比例

    # region 训练模型
    # 将训练数据馈送给模型
    # 模型学习将图像 train_images 和标签 train_labels 关联起来
    # 要求模型对测试集 test_images 进行预测
    # 验证预测是否与 test_labes 数组中的标签相匹配
    model.fit(train_images, train_labels, epochs=10)  # 开始训练，向模型馈送数据
    # endregion

    # 评估准确率
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)  # 比训练准确率低，在处理没有见过的输入，会表现为 过拟合

    # 对图像进行预测，附加一个 softmax 层，将模型的线性输出logits 转换成概率
    probability_model = tf.keras.Sequential([
        model,
        tf.keras.layers.Softmax()])

    predictions = probability_model.predict(test_images)  # 转换成概率
    print(np.argmax(predictions[0]))  # 第一个图片 返回给定轴概率最大元素的索引，索引即图片的类型 此处输出 9：短靴
    # print(np.argmax(predictions[0], axis=1))  # numpy 排序、条件筛选

    # region 查看 测试 第一个图片：应该是短靴
    # plt.figure()
    # plt.imshow(test_images[0])
    # plt.colorbar()
    # plt.grid(False)
    # plt.show()
    # endregion

    # DrawResult


if __name__ == '__main__':
    main()
