# ！/usr/bin/python3
# -*- coding: utf-8 -*-
# @TIme 2023/5/9 20:46
# @Author: SAM
# @File: overfit_and_underfit.py
# @Email: SAM-Turentu@outlook.com
# @Desc: 过拟合和欠拟合


import pathlib
import shutil
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow_docs import modeling, plots

from IPython import display

# pip install ipython
# pip install git+https://github.com/tensorflow/docs

from utils import join_path

logdir = pathlib.Path(tempfile.mkdtemp()) / "tensorboard_logs"
shutil.rmtree(logdir, ignore_errors=True)

# gz = tf.keras.utils.get_file('HIGGS.csv.gz', 'http://mlphysics.ics.uci.edu/data/higgs/HIGGS.csv.gz')  # 下载 超级慢
gz = join_path.keras_data_path.higgs_path

FEATURES = 28

# region 希格斯数据集

# CsvDataset 可直接从 Gzip文件读取csv，无需解压
ds = tf.data.experimental.CsvDataset(gz, [float(), ] * (FEATURES + 1), compression_type='GZIP')


def pack_row(*row):
    label = row[0]
    features = tf.stack(row[1:], 1)
    return features, label


# 创建一个数据集 以10000个为一批样本单位的批次，
packed_ds = ds.batch(10000).map(pack_row).unbatch()  # tf.data.Dataset

for features, label in packed_ds.batch(1000).take(1):
    print(features[0])
    plt.hist(features.numpy().flatten(), bins=101)
plt.show()

# 使用1000 个样本训练， 然后在用10,000个样本训练
N_VALIDATION = int(1e3)
N_TRAIN = int(1e4)
BUFFER_SIZE = int(1e4)
BATCH_SIZE = 500
STEPS_PER_EPOCH = N_TRAIN // BATCH_SIZE

validate_ds = packed_ds.take(N_VALIDATION).cache()  # 使用缓存，不用每次重新读取文件
train_ds = packed_ds.skip(N_VALIDATION).take(N_TRAIN).cache()

validate_ds = validate_ds.batch(BATCH_SIZE)  # batch 创建适当大小的批次进行训练
train_ds = train_ds.shuffle(BUFFER_SIZE).repeat().batch(BATCH_SIZE)

# endregion

# 深度学习模型更擅长拟合训练数据，真正的挑战是泛化而非拟合
# 模型 容量 大小要平衡，太大 容量过生，太小 容量不足；需要用不同的架构(层数活每层的正确大小)进行试验
# 先仅使用密集连接层 Dense 作为基线的简单模型开始，然后创建更大的模型并进行对比

# 演示拟合
# 减小学习率 使用schedules
lr_shedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    0.001,
    decay_steps=STEPS_PER_EPOCH * 1000,
    decay_rate=1,
    staircase=False)
# InverseTimeDecay 用于在 1,000 个周期时将学习率根据双曲线的形状降至基础速率的 1/2，在 2,000 个周期时将至 1/3

step = np.linspace(0, 100000)
lr = lr_shedule(step)
plt.figure(figsize=(8, 6))
plt.plot(step / STEPS_PER_EPOCH, lr)
plt.ylim([0, max(plt.ylim())])
plt.xlabel('Epoch')
_ = plt.ylabel('Learning Rate')
plt.show()


def get_optimizer():
    return tf.keras.optimizers.Adam(lr_shedule)


def get_callbacks(name):
    return [
        # tfdocs.modeling.EpochDots,  # 降低日志记录噪声，每个周期打印一个”.“  每隔100个周期打印一整套指标
        modeling.EpochDots(),  # 降低日志记录噪声，每个周期打印一个”.“  每隔100个周期打印一整套指标
        # 此回调是为了监视val_binary_crossentropy 而不是 val_loss
        tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy', patience=200),
        tf.keras.callbacks.TensorBoard(logdir / name)  # 为训练生成 TensorBoard 日志
    ]


def compile_and_fit(model, name, optimizer=None, max_epochs=10000):
    if optimizer is None:
        optimizer = get_optimizer()
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[
                      tf.keras.losses.BinaryCrossentropy(from_logits=True, name='binary_crossentropy'),
                      'accuracy'])

    model.summary()
    history = model.fit(
        train_ds,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=max_epochs,
        validation_data=validate_ds,
        callbacks=get_callbacks(name),
        verbose=0)
    return history


# 微模型
tiny_model = tf.keras.Sequential([
    layers.Dense(16, activation='elu', input_shape=(FEATURES,)),
    layers.Dense(1)
])

size_histories = {}

size_histories['Tiny'] = compile_and_fit(tiny_model, 'sizes/Tiny')

plotter = plots.HistoryPlotter(metric='binary_crossentropy', smoothing_std=10)
plotter.plot(size_histories)
plt.ylim([0.5, 0.7])
plt.show()


# 小模型
small_model = tf.keras.Sequential([
    # 设置两个隐层，每层16个单元
    layers.Dense(16, activation='elu', input_shape=(FEATURES,)),
    layers.Dense(16, activation='elu'),
    layers.Dense(1)
])

size_histories['Small'] = compile_and_fit(small_model, 'sizes/Small')


# 中等模型
medium_model = tf.keras.Sequential([
    layers.Dense(64, activation='elu', input_shape=(FEATURES,)),
    layers.Dense(64, activation='elu'),
    layers.Dense(64, activation='elu'),
    layers.Dense(1),
])
size_histories['Medium'] = compile_and_fit(medium_model, 'sizes/medium')


# 大模型
large_model = tf.keras.Sequential([
    layers.Dense(512, activation='elu', input_shape=(FEATURES,)),
    layers.Dense(512, activation='elu'),
    layers.Dense(512, activation='elu'),
    layers.Dense(512, activation='elu'),
    layers.Dense(1),
])
size_histories['large'] = compile_and_fit(large_model, 'sizes/large')

plotter = plots.HistoryPlotter(metric='binary_crossentropy', smoothing_std=10)
plotter.plot(size_histories)
plt.ylim([0.5, 0.7])
plt.show()