# -*- coding: utf-8 -*-
# Author: SAM
# Email: SAM-Turentu@outlook.com
# Name: LearnTF
# Filename: functional
# CreateTime: 2023/6/21 14:41
# Summary: 函数式API


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from utils import join_path
from utils.utils import tools

# region 简介

# 函数式 API 可以处理具有非线性拓扑的模型、具有共享层的模型，以及具有多个输入或输出的模型

# 数据的形状设置为784维向量，由于仅指定了每个样本的形状，因此始终忽略批次大小
inputs = keras.Input(shape=(784,))

img_inputs = keras.Input(shape=(32, 32, 3))

# 通过在此 inputs 对象上调用层，在层计算图中创建新的节点
dense = layers.Dense(64, activation='relu')
x = dense(inputs)

x = layers.Dense(64, activation='relu')(x)
outputs = layers.Dense(10)(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")

# 将模型绘制为计算图
keras.utils.plot_model(model, "functional/my_first_model.png")

# 绘制计算图中显示每层的输入和输出形状
keras.utils.plot_model(model, "functional/my_first_model_with_shape_info.png", show_shapes=True)

# endregion


# region 处理复杂的计算图拓扑


# region 具有多个输入和输出的模型

num_tags = 12
num_words = 10000
num_departments = 4

# 构建一个系统，该系统按照优先级对自定义问题工单进行排序，然后将工单传送到正确的部门，则此模型将具有三个输入
#   1. 工单标题（文本输入）
title_input = keras.Input(shape=(None,), name='title')
#   2. 工单的文本正文（文本输入）
body_input = keras.Input(shape=(None,), name='body')
#   3. 用户添加的任何标签
tags_input = keras.Input(shape=(num_tags,), name='tags')

title_features = layers.Embedding(num_words, 64)(title_input)
body_features = layers.Embedding(num_words, 64)(body_input)

title_features = layers.LSTM(128)(title_features)
body_features = layers.LSTM(128)(body_features)

x = layers.concatenate([title_features, body_features, tags_input])

# 此模型的两个输出
#   1. 介于 0 和 1 之间的优先级分数（标量 Sigmoid 输出）
priority_pred = layers.Dense(1, name='priority')(x)
#   2. 应该处理工单的部门（部门范围内的 Softmax 输出）
department_pred = layers.Dense(num_departments, name='department')(x)

model = keras.Model(
    inputs=[title_input, body_input, tags_input],  # 多个输入
    outputs=[priority_pred, department_pred]  # 多个输出
)

keras.utils.plot_model(model, "functional/multi_input_and_output_model.png", show_shapes=True)

# 为每个输出分配不同的损失。也可分配不同的权重
model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss=[
        keras.losses.BinaryCrossentropy(from_logits=True),
        keras.losses.CategoricalCrossentropy(from_logits=True)
    ],
    loss_weights=[1., 0.2]
)

# 输出层具有不同的名称，也可以使用对应的层名称指定损失 和 损失权重
# model.compile(
#     optimizer=keras.optimizers.RMSprop(1e-3),
#     loss={
#         'priority': keras.losses.BinaryCrossentropy(from_logits=True),
#         'department': keras.losses.CategoricalCrossentropy(from_logits=True),
#     },
#     loss_weights={'priority': 1., 'department': 0.2}
# )

# 输入数据
title_data = np.random.randint(num_words, size=(1280, 10))
body_data = np.random.randint(num_words, size=(1280, 100))
tags_data = np.random.randint(2, size=(1280, num_tags)).astype('float32')

# 目标数据
priority_targets = np.random.random(size=(1280, 1))
dept_targets = np.random.randint(2, size=(1280, num_departments))

model.fit(
    {'title': title_data, 'body': body_data, 'tags': tags_data},
    {'priority': priority_targets, 'department': dept_targets},
    epochs=2,
    batch_size=32
)

# endregion


# region 小ResNet模型

# 函数式API 还能容易的处理 非线性连接拓扑（这些模型的层没有按顺序连接）
#  常见用例：残差连接

#  为 CIFAR10 构建一个小 ResNet 模型以进行演示
inputs = keras.Input(shape=(32, 32, 3), name='img')
x = layers.Conv2D(32, 3, activation='relu')(inputs)
x = layers.Conv2D(64, 3, activation='relu')(x)
block_1_output = layers.MaxPooling2D(3)(x)

x = layers.Conv2D(64, 3, activation='relu', padding='same')(block_1_output)
x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
block_2_output = layers.add([x, block_1_output])

x = layers.Conv2D(64, 3, activation='relu', padding='same')(block_2_output)
x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
block_3_output = layers.add([x, block_2_output])

x = layers.Conv2D(64, 3, activation='relu')(block_3_output)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(10)(x)

model = keras.Model(inputs, outputs, name='toy_resnet')
model.summary()

keras.utils.plot_model(model, "functional/mini_resnet.png", show_shapes=True)

(x_train, y_train), (x_test, y_test) = tools.load_cifar10_data()
# (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['acc']
)

model.fit(x_train[:1000], y_train[:1000], batch_size=64, epochs=1, validation_split=0.2)

# endregion


# endregion


# region 共享层

shared_embedding = layers.Embedding(1000, 128)

text_input_a = keras.Input(shape=(None,), dtype='int32')

text_input_b = keras.Input(shape=(None,), dtype='int32')

# 重用同一层对两个输入进行编码
encoded_input_a = shared_embedding(text_input_a)
encoded_input_b = shared_embedding(text_input_b)

# endregion


# region 提取和重用层计算图中的节点

# path = 'D:\\Projects\\Python\\Local\\LearnTF\\keras_data\\VGG19\\vgg19_weights_tf_dim_ordering_tf_kernels.h5'

path = join_path.keras_data_path.vgg19_path

vgg19 = keras.applications.VGG19(weights=path)
# 'https://storage.googleapis.com/tensorflow/keras-applications/vgg19/vgg19_weights_tf_dim_ordering_tf_kernels.h5'

# 通过查询计算图数据结构获得的模型的中间激活
features_list = [layer.output for layer in vgg19.layers]

# 使用以下特征来创建新的特征提取模型，该模型会返回中间层激活的值
feat_extraction_model = keras.Model(inputs=vgg19.input, outputs=features_list)
img = np.random.random((1, 224, 224, 3)).astype("float32")
extracted_features = feat_extraction_model(img)

# endregion


# region 使用自定义层扩展 API


# endregion
