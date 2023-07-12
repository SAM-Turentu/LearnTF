# ！/usr/bin/python3
# -*- coding: utf-8 -*-
# @TIme 2023/7/12 16:37
# @Author: SAM
# @File: pandas_dataframe.py
# @Email: SAM-Turentu@outlook.com
# @Desc: 使用 tf.data 加载 pandas dataframes


import pandas as pd
import tensorflow as tf

# 使用 pandas 读取数据
path = '/Users/sam/projects/Python/oneself/tf_hw/load_data/heart/heart.csv'
df = pd.read_csv(path)
df.head()

# 将 thal 列（数据帧（dataframe）中的 object ）转换为离散数值
df['thal'] = pd.Categorical(df['thal'])
df['thal'] = df.thal.cat.codes

# 使用 tf.data.Dataset 读取数据
target = df.pop('target')
dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))
for feat, targ in dataset.take(5):
    print(f'Feature: {feat}, Target: {targ}')

# 随机读取 shuffle 并批量处理数据集
train_dataset = dataset.shuffle(len(df)).batch(1)


# 创建并训练模型
def get_compiled_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


model = get_compiled_model()
model.fit(train_dataset, epochs=15)

# region 代替特征列
# 将字典作为输入传输给模型就像创建 tf.keras.layers.Input 层的匹配字典一样简单，应用任何预处理并使用 functional api
#   可以使用它作为 feature columns 的替代方法
inputs = {key: tf.keras.layers.Input(shape=(), name=key) for key in df.keys()}
x = tf.stack(list(inputs.values()), axis=-1)

x = tf.keras.layers.Dense(10, activation='relu')(x)
output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model_func = tf.keras.Model(inputs=inputs, outputs=output)
model_func.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 与 tf.data 一起使用时，保存 pd.DataFrame 列结构的最简单方法是将 pd.DataFrame 转换为 dict ，并对该字典进行切片
dict_slices = tf.data.Dataset.from_tensor_slices((df.to_dict('list'), target.values)).batch(16)
for dict_slice in dict_slices.take(1):
    print(dict_slice)

model_func.fit(dict_slices, epochs=15)

# endregion
