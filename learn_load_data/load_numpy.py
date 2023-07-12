# ！/usr/bin/python3
# -*- coding: utf-8 -*-
# @TIme 2023/7/12 14:39
# @Author: SAM
# @File: load_numpy.py
# @Email: SAM-Turentu@outlook.com
# @Desc: 使用 tf.data 加载 NumPy 数据


import numpy as np
import tensorflow as tf
from utils import join_path

path = join_path.keras_data_path.helloworld_keras_mnist_data_path

with np.load(path) as data:
    train_examples = data['x_train']
    train_labels = data['y_train']
    test_examples = data['x_test']
    test_labels = data['y_test']

# 使用 tf.data.Dataset 加载 NumPy 数组
train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))

# region 使用该数据集


# region 打乱和批次化数据集
BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

# endregion


# region 建立和训练模型
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(
    optimizer=tf.keras.optimizers.RMSprop(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['sparse_categorical_accuracy']
)

model.compile(
    optimizer=tf.keras.optimizers.legacy.RMSprop(),  # M1 处理器使用旧版本
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['sparse_categorical_accuracy']
)

model.fit(train_dataset, epochs=10)
model.evaluate(test_dataset)

# endregion


# endregion
