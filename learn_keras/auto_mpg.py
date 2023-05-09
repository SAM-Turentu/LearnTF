# -*- coding: utf-8 -*-
# @File    :   auto_mpg.py
# @Time    :   2023/5/9
# @Author  :   SAM
# @Desc    :   回归


import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf

from tensorflow import keras

from utils import join_path


def main():
    # dataset_path = tf.keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")

    path = join_path.keras_data_path.auto_mpg_data_path
    columns = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']
    raw = pd.read_csv(path, names=columns, na_values='?', comment='\t', sep=' ', skipinitialspace=True)
    dataset = raw.copy()
    dataset.tail()
    # dataset.isna().sum()  # 字段为空的数据
    dataset = dataset.dropna()

    # Origin 需要转换为国家地区 1:USA,2:Europe,3:Japan
    origin = dataset.pop('Origin')  # 转换为下面的 独热码 （one-hot）
    dataset['USA'] = (origin == 1) * 1.0  # ===  df.loc[df['Origin'] == 1, 'USA'] = (df['Origin'] == 1) * 1.0
    dataset['Europe'] = (origin == 2) * 1.0
    dataset['Japan'] = (origin == 3) * 1.0

    # 将数据拆分，训练数据集，测试数据集
    train_dataset = dataset.sample(frac=0.8, random_state=0)  # 随机返回80%的数据，随机数种子random_state=1 抽样相同的结果，不设置，每次抽样结果不同
    test_dataset = dataset.drop(train_dataset.index)  # 训练集剩下的为测试集

    # 查看训练集中几队列的联合分布
    sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')
    # plt.savefig("./pairplot01.png")  # 需要手动另存为图片

    # 查看总体的数据统计
    train_stats = train_dataset.describe()
    train_stats.pop('MPG')
    train_stats = train_stats.transpose()

    # 将特征值从目标值活“标签”中分离。这个标签是使用训练模型进行预测的值
    train_labels = train_dataset.pop('MPG')
    test_labels = test_dataset.pop('MPG')

    # 数据规范化
    def norm(x):
        """标准差标准化 (x-均值) / 标准差"""
        return (x - train_stats['mean']) / train_stats['std']

    normed_train_data = norm(train_dataset)
    normed_test_data = norm(test_dataset)
    # 用于归一化输入的 数据统计(均值、标准差) 和 独热码one-hot 需要反馈给模型

    model = build_model(train_dataset)  # 构建 编译
    model.summary()

    # example_batch = normed_train_data[:10]
    # example_result = model.predict(example_batch)

    # 开启 1000个epochs
    EPOCHS = 1000

    # history = model.fit(
    #     normed_train_data, train_labels,
    #     epochs=EPOCHS, validation_split=0.2, verbose=0,
    #     callbacks=[PrintDot()])  # 回调

    # patience 值用来检查改进 epochs 数量
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)  # epochs 达到一定数量没有改进，自动停止训练
    history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                        validation_split=0.2, verbose=0, callbacks=[early_stop, PrintDot()])

    plot_history(history)

    # 使用测试集预测MPG值
    test_predictions = model.predict(normed_test_data).flatten()
    plt.scatter(test_labels, test_predictions)
    plt.xlabel('True Values [MPG]')
    plt.ylabel('Predictions [MPG]')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0, plt.xlim()[1]])
    plt.ylim([0, plt.ylim()[1]])
    _ = plt.plot([-100, 100], [-100, 100])
    plt.show()

    # 误差分布，不是完全的正态分布
    error = test_predictions - test_labels
    plt.hist(error, bins=25)
    plt.xlabel("Prediction Error [MPG]")
    _ = plt.ylabel("Count")
    plt.show()





# 构建模型
def build_model(train_dataset):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1),
    ])
    optimizer = tf.keras.optimizers.RMSprop(0.001)

    # 编译模型
    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model


# 通过为每个完成的使其打印一个点来显示训练进度
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print('')
        print('.', end='')


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mae'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mae'],
             label='Val Error')
    plt.ylim([0, 5])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mse'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'],
             label='Val Error')
    plt.ylim([0, 20])
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
