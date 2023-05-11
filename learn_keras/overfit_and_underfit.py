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

from IPython import display

# pip install ipython
# pip install git+https://github.com/tensorflow/docs
from utils import join_path


def main():
    logdir = pathlib.Path(tempfile.mkdtemp()) / "tensorboard_logs"
    shutil.rmtree(logdir, ignore_errors=True)

    # gz = tf.keras.utils.get_file('HIGGS.csv.gz', 'http://mlphysics.ics.uci.edu/data/higgs/HIGGS.csv.gz')  # 下载 超级慢
    gz = join_path.keras_data_path.higgs_path

    FEATURES = 28

    # region 希格斯数据集
    # CsvDataset 可直接从 Gzip文件读取csv，无需解压
    ds = tf.data.experimental.CsvDataset(gz, [float(), ] * (FEATURES + 1), compression_type='GZIP')

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


def pack_row(*row):
    label = row[0]
    features = tf.stack(row[1:], 1)
    return features, label


def get_optimizer(lr_shedule):
    return tf.keras.optimizers.Adam(lr_shedule)


if __name__ == '__main__':
    main()
