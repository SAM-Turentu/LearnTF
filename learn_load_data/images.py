# ！/usr/bin/python3
# -*- coding: utf-8 -*-
# @TIme 2023/6/7 16:17
# @Author: SAM
# @File: images.py
# @Email: SAM-Turentu@outlook.com
# @Desc: 加载和预处理图像


import os
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

image_count = len(list(data_dir.glob('*/*.jpg')))  # 文件夹中共有3670个图像

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

# region 显示训练集的前9个图像
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype('uint8'))
        plt.title(class_names[labels[i]])
        plt.axis('off')
plt.show()
# endregion

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


# region 使用 tf.data 进行更精细的控制
list_ds = tf.data.Dataset.list_files(str(data_dir / '*/*'), shuffle=False)
list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)

for f in list_ds.take(5):
    print(f.numpy())

class_names = np.array(sorted([item.name for item in data_dir.glob('*') if item.name != 'LICENSE.txt']))
print(class_names)

# 将数据集拆分为训练集和测试集
val_size = int(image_count * 0.2)
train_ds = list_ds.skip(val_size)
val_ds = list_ds.take(val_size)

# 每个数据集的长度
print(tf.data.experimental.cardinality(train_ds).numpy())
print(tf.data.experimental.cardinality(val_ds).numpy())


# 编写一个将文件路径转换为 (img, label) 对的短函数
def get_label(file_path):
    # 将路径转换为路径分量的列表
    parts = tf.strings.split(file_path, os.path.sep)
    one_hot = parts[-2] == class_names
    return tf.argmax(one_hot)


def decode_img(img):
    img = tf.io.decode_jpeg(img, channels=3)
    return tf.image.resize(img, [img_height, img_width])  # 调整图像大小


def process_path(file_path):
    label = get_label(file_path)
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label


# 使用 Dataset.map 创建 img, label对 的数据集
#   设置`num_parallel_calls`，使多个图像并行加载/处理
train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)

for image, label in train_ds.take(1):
    print('Image shape: ', image.numpy().shape)
    print('Label: ', label.numpy())


# region 训练的基本方法

def configure_for_performance(ds):
    # 数据充分打乱，分割为batch，永远重复
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


train_ds = configure_for_performance(train_ds)
val_ds = configure_for_performance(val_ds)

# endregion


# region 呈现数据

image_batch, labels_batch = next(iter(train_ds))

plt.figure(figsize=(10, 10))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image_batch[i].numpy().astype('uint8'))
    label = labels_batch[i]
    plt.title(class_names[label])
    plt.axis('off')
plt.show()

# endregion


# region 继续训练模型
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=3
)

# endregion


# endregion
