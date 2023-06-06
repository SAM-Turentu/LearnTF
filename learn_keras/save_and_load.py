# ！/usr/bin/python3
# -*- coding: utf-8 -*-
# @TIme 2023/6/6 09:27
# @Author: SAM
# @File: save_and_load.py
# @Email: SAM-Turentu@outlook.com
# @Desc: 保存和恢复模型


import os

import tensorflow as tf
from tensorflow import keras

from utils import join_path

# region 训练数据
file_path = join_path.keras_data_path.helloworld_keras_mnist_data_path  # 已经下载的数据集

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data(path=file_path)

# 使用前1000个样本
train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0


# endregion


# 构建一个简单的序列模型
def create_model():
    model = tf.keras.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10)
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    return model


# region 在训练期间保存模型（以 checkpoints 形式保存）

# 馈送数据，训练一次；回调 保存结果；

# checkpoint_path = 'training_1/cp.ckpt'
# checkpoint_dir = os.path.dirname(checkpoint_path)

# model = create_model()
#
# model.summary()

# # 创建一个只在训练期间保存权重的 tf.keras.callbacks.ModelCheckpoint 回调
# cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
#                                                  save_weights_only=True,
#                                                  verbose=1)

# model.fit(
#     train_images,
#     train_labels,
#     epochs=10,
#     validation_data=(test_images, test_labels),
#     callbacks=[cp_callback]
# )
#
# os.listdir(checkpoint_dir)


# endregion

# region Checkpoint 回调用法

# # 只要和训练是 model 相同的构架，就可以共享权重
# model = create_model()
#
# checkpoint_path = 'training_1/cp.ckpt'
# loss, acc = model.evaluate(test_images, test_labels, verbose=2)
# print(f'Untrained model, accuracy: {100 * acc:5.2f}%')  # 约10%
#
# # 从 checkpoint 加载权重并重新评估
# model.load_weights(checkpoint_path)
# loss, acc = model.evaluate(test_images, test_labels, verbose=2)
# print(f'Untrained model, accuracy: {100 * acc:5.2f}%')  # 约86%

# endregion


# region checkpoint 回调选项

checkpoint_path = 'training_2/cp-{epoch:04d}.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)

# region 训练 并 保存
# batch_size = 32
#
# # 每5个epochs 保存一次唯一命名的 checkpoint
# cp_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_path,
#     verbose=1,
#     save_weights_only=True,
#     save_freq=5 * batch_size
# )
#
# model = create_model()
# model.save_weights(checkpoint_path.format(epoch=0))
#
# model.fit(
#     train_images,
#     train_labels,
#     epochs=50,
#     batch_size=batch_size,
#     callbacks=[cp_callback],
#     validation_data=[test_images, test_labels],
#     verbose=0
# )

# endregion

# 检查生成的检查点并选择最新的检查点
os.listdir(checkpoint_dir)

latest = tf.train.latest_checkpoint(checkpoint_dir)
# 默认 tf 格式只保存最近的 5 个检查点

# 重置模型，并加载最新的检查点
model = create_model()

model.load_weights(latest)

loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'Restored model, accuracy:{100 * acc:5.2f}')

# endregion

# training_1  training_2 文件夹中
# 一个或多个包含模型权重的分片
# 一个索引文件，指示哪些权重存储在哪个分片中

# region 手动保存权重
model.save_weights('./checkpoints/my_checkpoint')

model = create_model()

model.load_weights('./checkpoints/my_checkpoint')

loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'Restored model, accuracy:{100 * acc:5.2f}')
# endregion


# region 保存整个模型
# tf.keras.Model.save 将模型的架构、权重、训练配置全部保存 在单个 file/folder 中。可以不使用python代码的情况下使用它
# 整个模型可以用两种不同的文件格式  SavedModel   HDF5
# tf2 默认 SavedModel

# region SavedModel 格式
# 以这种格式保存的可以 使用 tf.keras.models.load_model 还原
model = create_model()
model.fit(train_images, train_labels, epochs=5)
# !mkdir -p saved_model
model.save('saved_model/my_model')

# 加载
new_model = tf.keras.models.load_model('saved_model/my_model')
new_model.summary()

# 使用加载的模型进行评估预测
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'Restored model, accuracy:{100 * acc:5.2f}')

print(new_model.predict(test_images).shape)

# endregion

# region HDF5 格式, keras 使用 HDF5 标准提供了一种基本的保存格式

model = create_model()
model.fit(train_images, train_labels, epochs=5)
# 以后缀 .h5 结尾，就使用了 HDF5
model.save('my_model.h5')

new_model = tf.keras.models.load_model('my_model.h5')

new_model.summary()

loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'Restored model, accuracy:{100 * acc:5.2f}')

# endregion

# endregion


# 自定义对象的保存，SavedModel可以跳过下面的操作，使用HDF5 必须执行下面操作
#  在对象中定义一个 get_config 方法，并且可以选择定义一个 from_config 方法
#    get_config(self) 返回重新创建对象所需的参数的 json 可序列化字典
#    from_config(cls, config)  使用从 get_config 返回的配置来创建一个对象。默认情况，次还书将使用配置作为初始化 kwarg(return cls(**config))
#  加载模型是将对象传递给 custom_objects 参数
