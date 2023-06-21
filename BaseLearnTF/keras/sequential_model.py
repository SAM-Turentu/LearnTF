# -*- coding: utf-8 -*-
# Author: SAM
# Email: SAM-Turentu@outlook.com
# Name: LearnTF
# Filename: sequential_model
# CreateTime: 2023/6/20 17:51
# Summary: 模型


import tensorflow as tf
from tensorflow import keras

# region 设置

model = keras.Sequential([
    keras.layers.Dense(2, activation='relu', name='layer1'),
    keras.layers.Dense(3, activation='relu', name='layer2'),
    keras.layers.Dense(4, name='layer3'),
])

x = tf.ones((3, 3))
y = model(x)
# 上面等效于下面
layer1 = keras.layers.Dense(2, activation='relu', name='layer1')
layer2 = keras.layers.Dense(3, activation='relu', name='layer2')
layer3 = keras.layers.Dense(4, name='layer3')

y = tf.ones((3, 3))
y = layer3(layer2(layer1(x)))

# 在以下情况，Sequential model 不适用
#  1. 模型具有多个输入 或 多个输出
#  2. 任何图层都具有多个输入 或 多个输出
#  3. 需要执行图层共享
#  4. 需要非线性拓扑（例如残差连接、多分支模型）


# endregion


# region 创建 Sequential 模型

model = keras.Sequential([
    keras.layers.Dense(2, activation='relu'),
    keras.layers.Dense(3, activation='relu'),
    keras.layers.Dense(4),
])

# 或使用以增量方式创建顺序模型
model = keras.Sequential()
model.add(keras.layers.Dense(2, activation="relu"))
model.add(keras.layers.Dense(3, activation="relu"))
model.add(keras.layers.Dense(4))

# model.pop() # 删除图层

# endregion
