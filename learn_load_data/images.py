# ！/usr/bin/python3
# -*- coding: utf-8 -*-
# @TIme 2023/6/7 16:17
# @Author: SAM
# @File: images.py
# @Email: SAM-Turentu@outlook.com
# @Desc: 加载和预处理图像


import pathlib

import PIL.Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt

from utils import join_path

# 使用3种方式加载和预处理图像数据集
#  使用预处理效用函数 (例：tf.keras.utils.image_dataset_from_directory)  和层 (例：tf.keras.layers.Rescaling)  读取磁盘上的图像
#  使用 tf.data 从头编写自己的输入流水线
#  从 Tensorflow Datasets 中的大型目录下载数据集

# 下载数据
# dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
# data_dir = tf.keras.utils.get_file(origin=dataset_url, fname='flower_photos', untar=True)

# 数据路径（文件夹，非压缩文件）
data_dir = join_path.load_data.images_path

data_dir = pathlib.Path(data_dir)

# image_count = len(list(data_dir.glob('*/*.jpg')))  # 文件夹中共有3670个图像

# 玫瑰图片文件夹下所有文件的路径列表
# roses = list(data_dir.glob('roses/*'))
# PIL.Image.open(str(roses[0]))


batch_size = 32
img_height = 180
img_width = 180

# 验证拆分 80% 的图像用于训练
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=(img_height, img_width),  # 调整图像大小
    batch_size=batch_size  # 32个 图像
)
#  图像大小 也可以使用 tf.keras.layers.Resizing 层

# 20% 的图像用于验证
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = train_ds.class_names
print(train_ds.class_names)  # 在数据集中找到的类名   ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype('uint8'))
        plt.title(class_names[labels[i]])
        plt.axis('off')
plt.show()  # 显示训练集的前9个图像

for image_batch, labels_batch in train_ds:
    print(image_batch.shape)  # 打印张量: (32, 180, 180, 3)  # 这是由32个形状为 180 * 180 * 3 的突袭那个组成的批次，3：最后一个维度是指颜色通道RGB
    print(labels_batch.shape)  # 打印张量: (32,)  # 形状为  32 的张量，32个图像对应的标签
    break

# region 标准化数据
# RGB 值在 [0,255]之间。需要将值标准化为在 [0,1] 范围内
normalization_layer = tf.keras.layers.Rescaling(1. / 255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))  # 调用 Dataset.map 应用与数据集
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
print(np.min(first_image), np.max(first_image))  # 查看是否在 [0,1] 之间
# endregion

# region 配置数据集以提高性能（使用缓冲预获取）
#  在第一个周期期间加载图像后，Dataset.cache() 将图像保留在内存中。如果数据集太大，可以使用此方法创建高性能的磁盘缓存（数据缓存到磁盘上  tf.data  API 查看具体使用方法）
#  Dataset.prefetch() 会在训练时将 数据预处理和模型执行重叠

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
# endregion


# region 训练模型

num_classes = 5

model = tf.keras.Sequential([
    keras.layers.Rescaling(1. / 255),
    keras.layers.Conv2D(32, 3, activation='relu'),  # 卷积块，每个卷积块都有一个最大池化层
    keras.layers.MaxPooling2D(),  # 池化层
    keras.layers.Conv2D(32, 3, activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(32, 3, activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(num_classes)
])

model.compile(
    optimizer='adam',
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.fit(
    train_ds,  # 80% 数据集
    validation_data=val_ds,  # 20% 数据集
    epochs=3
)

# endregion
